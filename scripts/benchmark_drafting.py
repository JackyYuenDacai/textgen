"""Opt-in draft sweep through a local TextGen API; restores saved model settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ['fixed-3', 'fixed-2', 'fixed-4', 'fixed-5', 'adaptive-3', 'adaptive-4', 'adaptive-5']
PROMPTS = [
    ('coding', 'Write a complete Python LRU cache using a dictionary and a doubly linked list, without OrderedDict. Include get, put, deletion, invariants, and six unit tests. Explain its complexity.'),
    ('reasoning', 'A box contains 5 red, 7 blue, and 8 green balls. Draw 6 without replacement. Derive the exact probability of getting at least one of each color, using inclusion-exclusion and an independent counting argument. Show all calculations and a Python verification.'),
    ('long_context', '\n'.join(
        f'Record {i}: component=worker_{i % 31}, retry_limit={i % 5 + 1}, timeout_ms={100 + i % 900}, queue_limit={32 + i % 128}. Each worker must preserve FIFO ordering and log failed retries.'
        for i in range(450)) + '\nUsing these configuration records as context, design a robust Python asynchronous worker pool with bounded queues, graceful shutdown, retry handling, and tests. Explain deadlock risks and how to prevent them.'),
]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def summarize(samples):
    return dict(decode_tps=sum(s['new_tokens'] for s in samples) / sum(s['time_generate'] for s in samples),
                end_to_end_tps=sum(s['emitted_tokens'] for s in samples) / sum(s['request_seconds'] for s in samples))


def sweep(args):
    if not args.allow_model_reload:
        raise ValueError('Pass --allow-model-reload to authorize model reloads.')
    if urlsplit(args.api_base).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('This benchmark only supports a local TextGen server.')
    api_key = os.environ.get('TEXTGEN_BENCH_API_KEY', '')
    admin_key = os.environ.get('TEXTGEN_BENCH_ADMIN_KEY', api_key)

    def api(route, data=None):
        headers = {'Content-Type': 'application/json'}
        key = admin_key if route in ('internal/model/load', 'internal/model/unload') else api_key
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = Request(args.api_base.rstrip('/') + '/' + route,
                          data=json.dumps(data).encode() if data is not None else None, headers=headers)
        with urlopen(request, timeout=600) as response:
            return json.load(response)

    def info():
        return api('internal/model/info')

    original = info()
    if original['model_name'] != args.model or original['loader'] != 'ExLlamav3' or original.get('lora_names'):
        raise ValueError('Expected the requested ExLlamaV3 model without LoRAs to be loaded.')
    if not original.get('performance', {}).get('drafting'):
        raise ValueError('Restart TextGen with the adaptive-drafting update and load the model first.')
    recent = original['performance'].get('recent_requests', [])
    time.sleep(3)
    if recent != info()['performance'].get('recent_requests', []):
        raise ValueError('Server is generating. Wait for it to become idle before benchmarking.')
    import yaml
    saved_settings = yaml.safe_load((ROOT / 'user_data/models/config-user.yaml').read_text(encoding='utf-8')) or {}
    restore_args = {}
    for pattern, values in saved_settings.items():
        if re.match(pattern.lower(), args.model.lower()):
            restore_args.update(values)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(model_name=args.model, args=restore_args)
    save(output / 'restore.json', config)
    save(output / 'previous-performance.json', original)
    reports = []

    def generate(name, question, budget, repeat):
        before = info()['performance'].get('recent_requests', [])
        payload = dict(model=args.model, messages=[dict(role='user', content=question)],
                       temperature=0, seed=42, max_tokens=budget, enable_thinking=False, stream=False)
        start = time.perf_counter()
        response = api('chat/completions', payload)
        elapsed = time.perf_counter() - start
        after = info()
        records = after['performance']['recent_requests']
        previous_id = before[-1]['job_id'] if before else -1
        new = [s for s in records if s['job_id'] > previous_id]
        if after['model_name'] != args.model or len(new) != 1 or not new[0].get('completed'):
            raise RuntimeError('Concurrent activity or incomplete generation invalidated this measurement.')
        answer = response['choices'][0]['message'].get('content') or ''
        return dict(prompt_name=name, repeat=repeat, prompt=question, answer=answer,
                    answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                    request_seconds=elapsed, finish_reason=response['choices'][0]['finish_reason'], **new[0])

    try:
        for name in args.profiles:
            strategy, length = name.rsplit('-', 1)
            length, adaptive = int(length), strategy == 'adaptive'
            print(f'Loading {name}. Do not use chat or other API clients during the comparison.', flush=True)
            profile_args = dict(restore_args, draft_max=length, exl3_dynamic_draft=adaptive, exl3_draft_confidence=0.4)
            api('internal/model/load', dict(model_name=args.model, args=profile_args))
            drafting = info()['performance']['drafting']
            if drafting['mode'] == 'none' or drafting['adaptive'] != adaptive or drafting['max_tokens'] != length:
                raise RuntimeError(f'The requested drafting profile was not activated: {drafting}')
            report = dict(model=args.model, profile=dict(length=length, adaptive=adaptive, confidence=0.4),
                          args=profile_args, drafting=drafting, temperature=0, seed=42, thinking=False,
                          max_new_tokens=512, protocol='Three 128-token warmups; three prompts, two measured rounds; EOS allowed.', samples=[])
            for prompt_name, question in PROMPTS:
                warmup = generate(prompt_name, question, 128, -1)
                print(f"  warmup {prompt_name}: {warmup['decode_tokens_per_second']:.2f} tok/s", flush=True)
            for repeat in range(2):
                for prompt_name, question in PROMPTS:
                    sample = generate(prompt_name, question, 512, repeat)
                    report['samples'].append(sample)
                    save(output / f'{name}.json', report)
                    print(f"  {prompt_name} round={repeat + 1}: {sample['decode_tokens_per_second']:.2f} tok/s, "
                          f"prompt={sample['prompt_tokens']}, acceptance={sample.get('draft_acceptance', 0):.1%}", flush=True)
            report.update(summarize(report['samples']), status='completed')
            save(output / f'{name}.json', report)
            reports.append({k: v for k, v in report.items() if k != 'samples'})
            save(output / 'summary.json', reports)
            print(f"{name}: {report['decode_tps']:.2f} decode tok/s", flush=True)
    finally:
        print('Restoring the original saved model settings...', flush=True)
        api('internal/model/load', config)
        if info()['model_name'] != args.model:
            raise RuntimeError('Model restoration failed. Reload using restore.json.')
        print('Original model restored.', flush=True)
    print(f'Reports: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default='http://127.0.0.1:5000/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=PROFILES)
    parser.add_argument('--allow-model-reload', action='store_true')
    sweep(parser.parse_args())
