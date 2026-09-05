"""Run pinned Inspect Evals tasks in the dedicated benchmark environment."""

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

PREFIX = '@@TEXTGEN_BENCHMARK '


def prepare_nltk():
    """Load IFEval's English sentence tables into the benchmark cache.

    The upstream helper ignores NLTK_DATA and always downloads to the home
    directory. Fetch the fixed official resource archive and only its four
    English text tables; no pickles, arbitrary URLs, or archive path extraction.
    """
    import io
    import urllib.request
    import zipfile

    import nltk

    root = Path(os.environ['NLTK_DATA'])
    files = ('collocations.tab', 'sent_starters.txt', 'abbrev_types.txt', 'ortho_context.tab')
    destination = root / 'tokenizers' / 'punkt_tab' / 'english'
    if not all((destination / name).is_file() for name in files):
        url = 'https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip'
        print('Downloading pinned English sentence-tokenizer tables for IFEval…', flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            archive_bytes = response.read(8 * 1024 * 1024)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            destination.mkdir(parents=True, exist_ok=True)
            for name in files:
                (destination / name).write_bytes(archive.read('punkt_tab/english/' + name))
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    # Exercise the actual scorer dependency before sending any model requests.
    nltk.sent_tokenize('One sentence. Another sentence.')


def emit(**event):
    print(PREFIX + json.dumps(event, ensure_ascii=False), flush=True)


def check():
    from inspect_ai import eval_async  # noqa: F401
    from inspect_evals.gsm8k import gsm8k  # noqa: F401
    from inspect_evals.humaneval import humaneval  # noqa: F401
    from inspect_evals.ifeval import ifeval  # noqa: F401
    import instruction_following_eval  # noqa: F401
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI  # noqa: F401
    print('Inspect AI and all three benchmark tasks are installed.', flush=True)


def make_task(config):
    if config['task'] == 'gsm8k':
        from inspect_evals.gsm8k import gsm8k
        return gsm8k(fewshot=0)
    if config['task'] == 'ifeval':
        from inspect_evals.ifeval import ifeval
        from unittest.mock import patch

        with patch('instruction_following_eval.evaluation.ensure_nltk_resource', prepare_nltk):
            return ifeval()
    if config['task'] == 'humaneval':
        from inspect_evals.humaneval import humaneval
        # Never fall back to executing generated Python on the host.
        result = subprocess.run(['docker', 'info'], capture_output=True, timeout=20)
        if result.returncode:
            raise RuntimeError('HumanEval requires a running Docker engine with Linux containers. Start Docker Desktop and retry.')
        return humaneval(sandbox='docker')
    raise ValueError('Unsupported benchmark task.')


def sample_summary(sample):
    return {
        'id': str(sample.id),
        'input': sample.input,
        'target': sample.target,
        'answer': sample.output.completion if sample.output else '',
        'scores': {name: value.model_dump(mode='json') for name, value in (sample.scores or {}).items()},
        'error': sample.error.message if sample.error else None,
        'truncated': any(choice.stop_reason == 'max_tokens' for choice in sample.output.choices) if sample.output else False,
    }


def metric_rows(log):
    rows = []
    for score in log.results.scores if log.results else []:
        for name, metric in score.metrics.items():
            # IFEval may expose a dictionary of strict/loose instruction/prompt metrics.
            value = metric.value
            if isinstance(value, dict):
                rows.extend({'scorer': score.name, 'metric': key, 'value': val} for key, val in value.items())
            else:
                rows.append({'scorer': score.name, 'metric': name, 'value': value})
    return rows


async def evaluate(run_dir, config):
    from inspect_ai import eval_async
    from inspect_ai.hooks import Hooks, hooks

    @hooks(name='textgen-benchmark-progress', description='Report sample progress to the TextGen web UI')
    class Progress(Hooks):
        async def on_sample_end(self, data):
            emit(kind='sample', sample=sample_summary(data.sample))

    emit(status='loading', message='Loading the benchmark dataset (downloaded on first use).')
    task = make_task(config)
    total = min(config['limit'], len(task.dataset))
    emit(status='running', total=total, message='Evaluating one sample at a time.',
         versions={p: importlib.metadata.version(p) for p in ('inspect-ai', 'inspect-evals')},
         dataset_size=len(task.dataset), protocol='zero-shot, one generation per sample, seeded shuffled subset')
    extra = {}
    if config['thinking'] != 'Server default':
        extra['enable_thinking'] = config['thinking'] == 'Enabled'
    os.environ['TEXTGEN_API_KEY'] = os.environ.pop('TEXTGEN_BENCH_API_KEY', 'textgen-local')
    evaluation = asyncio.create_task(eval_async(
        task, model='openai-api/textgen/' + config['model'], model_base_url=config['endpoint'],
        model_args={'responses_api': False, 'stream': False},
        log_dir=str(run_dir / 'logs'), log_format='json',
        limit=config['limit'], sample_shuffle=config['seed'], epochs=1,
        max_samples=1, max_connections=1, max_sandboxes=1, fail_on_error=True,
        max_tokens=config['max_tokens'], temperature=config['temperature'], seed=config['seed'],
        extra_body=extra or None, timeout=600, max_retries=0,
        log_buffer=1, log_model_api=False, sandbox_cleanup=True, ctl_server=False,
    ))
    cancelled = False
    while not evaluation.done():
        if (run_dir / 'stop').exists():
            cancelled = True
            evaluation.cancel()
            break
        await asyncio.sleep(0.5)
    try:
        logs = await evaluation
    except asyncio.CancelledError:
        emit(status='cancelled', message='Evaluation cancelled. Completed sample records are retained.')
        return
    if cancelled:
        emit(status='cancelled', message='Evaluation cancelled. Completed sample records are retained.')
        return
    if not logs:
        emit(status='error', message='Inspect returned no evaluation logs. See runner output for details.')
        return
    log = logs[0]
    status = 'cancelled' if cancelled or log.status == 'cancelled' else ('completed' if log.status == 'success' else 'error')
    emit(status=status, metrics=metric_rows(log),
         message=log.error.message if log.error else 'Evaluation finished. Scores apply to the sampled protocol.',
         results=log.results.model_dump(mode='json') if log.results else None)


if __name__ == '__main__':
    if sys.argv[1:] == ['--check']:
        check()
    else:
        try:
            directory = Path(sys.argv[1]).resolve()
            config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            asyncio.run(evaluate(directory, config))
        except Exception as error:
            emit(status='error', message=f'{type(error).__name__}: {error}')
            sys.exit(1)
