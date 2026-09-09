"""Explicitly requested live full-context test, using synthetic text only.

Runs generation on the already loaded server. Does not reload or change its
settings. This populates/replaces prefix cache entries as ordinary requests do.
"""
import hashlib
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

from tokenizers import Tokenizer

BASE = 'http://127.0.0.1:5000'
OUT = Path('user_data/logs/full_context_live_' + time.strftime('%Y%m%d_%H%M%S'))


def api(path, body=None, timeout=600):
    request = urllib.request.Request(BASE + path,
        data=None if body is None else json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = api('/v1/internal/model/info', timeout=10)
    recent = before['performance']['recent_requests']
    configuration = recent[-1]['configuration']
    capacity = configuration['context_capacity']
    headroom = 1 + before['performance']['drafting']['max_tokens']
    session = before['performance']['history']['session_id']
    model = before['model_name']
    (OUT / 'before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    tokenizer = Tokenizer.from_file(str(Path('user_data/models') / model / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.encode_special_tokens = False
    header = 'Synthetic context stress test. The following records are test data.\n'
    records = ''.join(f'Record {i:05d}: key={hashlib.sha256(str(i).encode()).hexdigest()[:24]}; '
                      f'value={i % 997}; sample text 中文 memory check.\n' for i in range(14000))
    suffix = '\nEnd of data. Continue the integer sequence for as long as possible: 1, 2, 3,'
    ids = tokenizer.encode(header + records, add_special_tokens=False).ids

    def prompt_for(target):
        # Leave a short margin for suffix/BOS, then fill measured spare tokens.
        text = tokenizer.decode(ids[:target - 80], skip_special_tokens=False)
        candidate = text + suffix
        count = api('/v1/internal/token-count', {'text': candidate})['length']
        for _ in range(5):
            if count == target:
                return candidate
            if count > target:
                raise RuntimeError(f'Prompt exceeded target: {count} > {target}')
            text += ' x' * (target - count)
            candidate = text + suffix
            count = api('/v1/internal/token-count', {'text': candidate})['length']
        raise RuntimeError(f'Could not fit exact token target {target}: {count}')

    target = capacity - headroom - 128
    prompt = prompt_for(target)
    summary = {'model': model, 'configuration': configuration, 'headroom': headroom,
               'prompt_tokens': target, 'runs': [], 'output_directory': str(OUT)}
    print(json.dumps({'prepared': summary}), flush=True)

    def run(label, text, expected_tokens, max_tokens):
        current = api('/v1/internal/model/info', timeout=10)
        if current['performance']['history']['session_id'] != session:
            raise RuntimeError('Loaded model session changed; stopping test')
        start_sequence = current['performance']['history']['total_recorded']
        body = {'model': model, 'prompt': text, 'max_tokens': max_tokens, 'temperature': 0,
                'seed': 908100 + len(summary['runs']), 'stream': False, 'ban_eos_token': True}
        samples = []
        started = time.perf_counter()
        print(json.dumps({'started': label, 'prompt_tokens': expected_tokens,
                          'requested_output': max_tokens}), flush=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(api, '/v1/completions', body)
            while True:
                try:
                    response = request.result(timeout=5)
                    break
                except TimeoutError:
                    sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                             '--format=csv,noheader,nounits'], capture_output=True,
                                            text=True, timeout=5)
                    samples.append({'seconds': round(time.perf_counter() - started, 2),
                                    'gpu': sample.stdout.strip()})
                    print(json.dumps({'running': label, **samples[-1]}), flush=True)
        after = api('/v1/internal/model/info', timeout=10)
        if after['performance']['history']['session_id'] != session:
            raise RuntimeError('Loaded model session changed during request')
        jobs = [job for job in after['performance']['recent_requests']
                if job['sequence'] > start_sequence and job.get('prompt_tokens') == expected_tokens]
        result = {'label': label, 'elapsed_seconds': time.perf_counter() - started,
                  'response': response, 'jobs': jobs, 'gpu_samples': samples}
        (OUT / (label + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
        if len(jobs) != 1 or not jobs[0]['completed']:
            raise RuntimeError('Could not identify one successfully completed test job')
        job = jobs[0]
        if job['prompt_budget']['loader_dropped_tokens'] != 0:
            raise RuntimeError('Server truncated the test prompt')
        summary['runs'].append({key: job.get(key) for key in (
            'prompt_tokens', 'cached_tokens', 'new_tokens', 'time_prefill', 'time_generate',
            'decode_tokens_per_second', 'prompt_budget', 'memory_pressure', 'memory_after')})
        summary['runs'][-1]['label'] = label
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'finished': label, 'prefill': job['time_prefill'],
                          'cached_tokens': job['cached_tokens'], 'new_tokens': job['new_tokens'],
                          'decode_tokens_per_second': job['decode_tokens_per_second']}), flush=True)

    run('cold_full', prompt, target, 128)
    run('warm_full', prompt, target, 128)
    boundary_target = capacity - headroom - 1
    # Append one-token fillers, preserving all earlier prompt tokens for reuse.
    boundary = prompt + ' x' * (boundary_target - target)
    measured = api('/v1/internal/token-count', {'text': boundary})['length']
    if measured != boundary_target:
        raise RuntimeError(f'Boundary count mismatch: {measured} != {boundary_target}')
    run('exact_boundary', boundary, boundary_target, 128)
    print(json.dumps({'result': 'passed', 'output_directory': str(OUT)}), flush=True)


if __name__ == '__main__':
    main()
