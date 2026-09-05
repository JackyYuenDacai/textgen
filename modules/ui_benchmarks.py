"""Native Gradio controls for Inspect AI capability evaluations."""

import html
from functools import lru_cache

import gradio as gr

from modules import shared
from modules.capability_benchmarks import BenchmarkManager, TASKS, validate_config


@lru_cache(maxsize=1)
def manager():
    return BenchmarkManager(shared.user_data_dir / 'benchmarks')


def require_owner():
    if shared.args.multi_user:
        raise gr.Error('Benchmark execution and saved reports are disabled in multi-user mode.')


def install():
    require_owner()
    return manager().install()


def start(task, endpoint, model, api_key, limit, max_tokens, temperature, seed, thinking):
    require_owner()
    try:
        config = validate_config(task, endpoint, model, limit, max_tokens, temperature, seed, thinking)
        return manager().start(config, api_key.strip())
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error


def stop():
    require_owner()
    return manager().stop()


def render_record(record):
    if not record:
        return '<div class="bench-empty">Choose a benchmark and start an evaluation. Results will appear here.</div>'
    esc = lambda value: html.escape(str(value))
    config = record['config']
    total = max(record.get('total', config['limit']), 1)
    done = record.get('completed', 0)
    status = record['status']
    # A server restart cannot resurrect a Python worker from the previous server.
    if status in ('starting', 'loading', 'running') and (not manager().active or manager().active['id'] != record['id']):
        status = 'interrupted'
    cards = ''.join(f'<div class="bench-stat"><span>{label}</span><strong>{esc(value)}</strong></div>' for label, value in [
        ('Status', status), ('Processed', f'{done} / {total}'),
        ('Errors', record.get('errors', 0)), ('Token-limit hits', record.get('truncated', 0)),
    ])
    metrics = []
    for metric in record.get('metrics', []):
        name, value = metric['metric'], metric['value']
        is_accuracy = name == 'accuracy' or name.endswith('_acc')
        shown = f'{value:.1%}' if is_accuracy and isinstance(value, (int, float)) else str(value)
        bar = f'<progress max="1" value="{max(0, min(1, value))}"></progress>' if is_accuracy and isinstance(value, (int, float)) else ''
        metrics.append(f'<tr><td>{esc(metric["scorer"])}</td><td>{esc(name)}</td><td>{esc(shown)}{bar}</td></tr>')
    table = '<table><thead><tr><th>Scorer</th><th>Metric</th><th>Result</th></tr></thead><tbody>' + ''.join(metrics) + '</tbody></table>' if metrics else '<p>Final scores appear when evaluation finishes. Completed answers are retained as the run progresses.</p>'
    return f'''<section class="bench-report">
        <h2>{esc(TASKS[config['task']])}</h2>
        <p>{esc(config['model'])} · {esc(record['id'])}</p>
        <div class="bench-stats">{cards}</div>
        <progress class="bench-progress" max="{total}" value="{done}" aria-label="Samples processed"></progress>
        <p>{esc(record.get('message', 'Starting evaluation…'))}</p>
        {table}
        <p class="bench-protocol">Subset limit {config['limit']} · seed {config['seed']} · temperature {config['temperature']}
        · output budget {config['max_tokens']} · thinking: {esc(config['thinking'])}.
        One generation per sample. Scores from partial runs or custom settings are not official leaderboard results.</p>
    </section>'''


def live_report():
    if shared.args.multi_user:
        return ''
    instance = manager()
    setup = 'Installing…' if instance.setup_running else ('Ready' if instance.ready else 'Not installed')
    with instance.lock:
        tail = html.escape('\n'.join(instance.console))
    return f'<p class="bench-environment">Evaluation environment: {setup}</p>' + render_record(instance.read()) + (
        f'<details class="bench-console"><summary>Runner / setup output</summary><pre>{tail}</pre></details>' if tail else '')


def refresh_runs():
    require_owner()
    ids = [p.parent.name for p in manager().runs()]
    return gr.update(choices=ids, value=ids[0] if ids else None)


def load_run(run_id):
    require_owner()
    record = manager().read(run_id) if run_id else None
    choices = [(f'{index + 1}: {sample["id"]}', str(index)) for index, sample in enumerate(record.get('samples', []))] if record else []
    return render_record(record), gr.update(choices=choices, value=choices[0][1] if choices else None), None


def load_sample(run_id, index):
    require_owner()
    record = manager().read(run_id) if run_id else None
    if not record or index is None:
        return None
    try:
        return manager().sample(run_id, int(index))
    except (ValueError, IndexError, OSError):
        return None


def export(run_id):
    require_owner()
    if not run_id:
        raise gr.Error('Select a run first.')
    try:
        return manager().export(run_id)
    except ValueError as error:
        raise gr.Error(str(error)) from error


def create_ui():
    with gr.Tab('Benchmarks', elem_id='benchmarks-tab'):
        gr.Markdown('## Capability benchmarks\nEvaluate maths, Python programming, and instruction following using Inspect AI. '
                    'Datasets download on first use. Start with 10 samples to check your settings, then increase the sample count.')
        if shared.args.multi_user:
            gr.Markdown('Benchmark execution and saved reports are disabled in multi-user mode.')
            return
        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(choices=[(label, key) for key, label in TASKS.items()], value='gsm8k', label='Benchmark')
                gr.Markdown('GSM8K uses zero-shot answer matching. HumanEval checks generated Python against unit tests in **Docker Linux containers**. '
                            'IFEval reports strict/loose instruction and prompt accuracy. No judge model is required.')
                endpoint = gr.Textbox(value=f'http://127.0.0.1:{shared.args.api_port}/v1', label='OpenAI-compatible API base URL',
                                      info='Enable TextGen’s API in Session, or supply another endpoint. Requests run on the server.')
                model = gr.Textbox(value=lambda: shared.model_name if shared.model_name != 'None' else '', label='Model ID')
                gr.Button('Use loaded model').click(lambda: shared.model_name, outputs=model, api_name=False, queue=False)
                api_key = gr.Textbox(value='', type='password', label='API key (optional)', info='Held for this run only; not included in saved settings.')
                with gr.Row():
                    limit = gr.Number(value=10, minimum=1, maximum=10000, precision=0, label='Sample limit')
                    seed = gr.Number(value=42, minimum=0, maximum=2147483647, precision=0, label='Sampling / generation seed')
                with gr.Row():
                    tokens = gr.Number(value=8192, minimum=128, maximum=65536, precision=0, label='Max output tokens per sample')
                    temperature = gr.Number(value=0, minimum=0, maximum=2, label='Temperature')
                thinking = gr.Dropdown(choices=['Server default', 'Enabled', 'Disabled'], value='Server default', label='Thinking mode',
                                      info='Enabled/Disabled sends TextGen’s enable_thinking option; use Server default for other providers.')
                gr.Markdown('Runs submit one request at a time and share model capacity with chat. Changing the loaded model or its template during a run invalidates comparisons. '
                            'Remote endpoints may charge for requests; benchmark prompts are sent to the URL above.')
                with gr.Row():
                    run = gr.Button('Start benchmark', variant='primary')
                    cancel = gr.Button('Stop benchmark')
                with gr.Accordion('Evaluation environment', open=False):
                    gr.Markdown('Install pinned Inspect AI / Inspect Evals dependencies into `user_data/benchmarks/env`. '
                                'This requires internet access. HumanEval also needs Docker Desktop running with Linux containers.')
                    setup = gr.Button('Install / repair benchmark environment')
                action_status = gr.Textbox(label='Action status', interactive=False)
            with gr.Column(scale=2):
                gr.HTML(value=live_report, every=2, elem_id='benchmarks-live')
        with gr.Accordion('Saved runs and individual answers', open=True):
            with gr.Row():
                saved = gr.Dropdown(choices=[], label='Saved run')
                refresh = gr.Button('Refresh saved runs')
                download = gr.Button('Export selected run')
            historical = gr.HTML()
            sample = gr.Dropdown(choices=[], label='Sample to inspect')
            detail = gr.JSON(label='Question, expected answer, model response, and scoring details')
            artifact = gr.File(label='Download report and Inspect logs', interactive=False)
        run.click(start, [task, endpoint, model, api_key, limit, tokens, temperature, seed, thinking], action_status, api_name=False, queue=False)
        cancel.click(stop, outputs=action_status, api_name=False, queue=False)
        setup.click(install, outputs=action_status, api_name=False, queue=False)
        refresh.click(refresh_runs, outputs=saved, api_name=False, queue=False)
        saved.change(load_run, saved, [historical, sample, detail], api_name=False, queue=False)
        sample.change(load_sample, [saved, sample], detail, api_name=False, queue=False)
        download.click(export, saved, artifact, api_name=False, queue=False)
