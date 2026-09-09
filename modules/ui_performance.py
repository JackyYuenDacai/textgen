import html
import math
from datetime import datetime

from modules import shared


REFRESH_SECONDS = 2


def _number(value):
    """Return a finite float, or None for missing/invalid metric values."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _rate(record, key='decode_tokens_per_second'):
    if key == 'end_to_end':
        tokens = _number(record.get('new_tokens'))
        seconds = _number(record.get('total_seconds'))
        return tokens / seconds if tokens is not None and seconds and seconds > 0 else None

    return _number(record.get(key))


def _format_number(value, digits=2, suffix='', empty='—'):
    value = _number(value)
    return f'{value:,.{digits}f}{suffix}' if value is not None else empty


def _format_integer(value):
    value = _number(value)
    return f'{int(value):,}' if value is not None else '—'


def _format_percent(value):
    value = _number(value)
    return f'{value * 100:.1f}%' if value is not None else '—'


def _metric_card(label, value, detail, tone='blue'):
    return f'''
        <article class="perf-metric perf-tone-{tone}">
            <div class="perf-metric-label">{html.escape(label)}</div>
            <div class="perf-metric-value">{html.escape(value)}</div>
            <div class="perf-metric-detail">{html.escape(detail)}</div>
        </article>
    '''


def _line_chart(records):
    completed = [record for record in records if record.get('completed')]
    series = [
        ('Decode', '#4a72ff', [_rate(record) for record in completed]),
        ('End-to-end', '#21a179', [_rate(record, 'end_to_end') for record in completed]),
    ]
    values = [value for _, _, points in series for value in points if value is not None]
    if not values:
        return '<div class="perf-chart-empty">Complete a generation to populate this chart.</div>'

    width, height = 760, 260
    left, right, top, bottom = 54, 18, 22, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(max(values) * 1.12, 1)
    count = max(len(completed), 1)

    def point(index, value):
        x = left + (plot_width / (count - 1) * index if count > 1 else plot_width / 2)
        y = top + plot_height - (value / y_max * plot_height)
        return x, y

    grid = []
    for index in range(5):
        value = y_max * (4 - index) / 4
        y = top + plot_height * index / 4
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="perf-grid-line" />'
            f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="perf-axis-label">{value:.0f}</text>'
        )

    paths = []
    for label, color, values_for_series in series:
        segments = []
        current = []
        for index, value in enumerate(values_for_series):
            if value is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            current.append(point(index, value))
        if current:
            segments.append(current)

        for segment in segments:
            if len(segment) == 1:
                x, y = segment[0]
                paths.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
            else:
                points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in segment)
                paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />')

        valid_points = [(index, value) for index, value in enumerate(values_for_series) if value is not None]
        if valid_points:
            index, value = valid_points[-1]
            x, y = point(index, value)
            paths.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="var(--bg-elevated)" stroke-width="2" />')

    first_job = html.escape(str(completed[0].get('job_id', '—')))
    last_job = html.escape(str(completed[-1].get('job_id', '—')))
    legend = ''.join(
        f'<span><i style="background:{color}"></i>{label}</span>' for label, color, _ in series
    )
    return f'''
        <div class="perf-chart-legend">{legend}</div>
        <svg class="perf-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Decode and end-to-end token throughput by request">
            {''.join(grid)}
            <text x="13" y="{top + plot_height / 2:.1f}" transform="rotate(-90 13 {top + plot_height / 2:.1f})" text-anchor="middle" class="perf-axis-title">tokens / second</text>
            {''.join(paths)}
            <text x="{left}" y="{height-10}" text-anchor="start" class="perf-axis-label">job {first_job}</text>
            <text x="{width-right}" y="{height-10}" text-anchor="end" class="perf-axis-label">job {last_job}</text>
        </svg>
    '''


def _latency_chart(records):
    completed = [record for record in records if record.get('completed')][-16:]
    rows = []
    for record in completed:
        prefill = _number(record.get('time_prefill')) or 0
        decode = _number(record.get('time_generate')) or 0
        total = prefill + decode
        if total <= 0:
            continue
        rows.append((record, prefill, decode, total))

    if not rows:
        return '<div class="perf-chart-empty">Latency details are not available yet.</div>'

    max_total = max(total for _, _, _, total in rows)
    bars = []
    for record, prefill, decode, total in rows:
        prefill_width = prefill / max_total * 100
        decode_width = decode / max_total * 100
        job_id = html.escape(str(record.get('job_id', '—')))
        title = html.escape(f'Job {job_id}: prefill {prefill:.3f}s, decode {decode:.3f}s')
        bars.append(f'''
            <div class="perf-latency-row" title="{title}">
                <span class="perf-latency-job">{job_id}</span>
                <div class="perf-latency-track">
                    <i class="perf-latency-prefill" style="width:{prefill_width:.3f}%"></i>
                    <i class="perf-latency-decode" style="width:{decode_width:.3f}%"></i>
                </div>
                <span class="perf-latency-value">{total:.2f}s</span>
            </div>
        ''')

    return f'''
        <div class="perf-chart-legend">
            <span><i style="background:#f0a33a"></i>Prefill</span>
            <span><i style="background:#7b61d1"></i>Decode</span>
        </div>
        <div class="perf-latency-list">{''.join(bars)}</div>
    '''


def _memory_label(record, field):
    devices = (record.get('memory_after') or {}).get('devices') or []
    labels = []
    for device in devices:
        value = _number(device.get(field))
        if value is not None:
            labels.append(f"GPU {device.get('device')}: {value / 1024**2:,.0f} MiB")
    return html.escape('; '.join(labels) or 'N/A')


def _history_table(records):
    rows = []
    for record in reversed(records):
        completed = bool(record.get('completed'))
        status = 'Complete' if completed else 'Stopped'
        status_class = 'complete' if completed else 'stopped'
        job_id = html.escape(str(record.get('job_id', '—')))
        rows.append(f'''
            <tr>
                <td class="perf-job">#{job_id}</td>
                <td><span class="perf-status perf-status-{status_class}">{status}</span></td>
                <td>{_format_number(_rate(record), suffix=' tok/s')}</td>
                <td>{_format_number(_rate(record, 'end_to_end'), suffix=' tok/s')}</td>
                <td>{_format_number(record.get('time_prefill'), 3, 's')}</td>
                <td>{_format_number(record.get('time_to_first_output'), 3, 's')}</td>
                <td>{_format_integer(record.get('new_tokens', record.get('emitted_tokens')))}</td>
                <td>{_format_integer(record.get('prompt_tokens'))}</td>
                <td>{_format_integer(record.get('cached_tokens'))}</td>
                <td>{_format_percent(record.get('draft_acceptance'))}</td>
                <td>{_format_integer((record.get('configuration') or {}).get('max_chunk_size'))}</td>
                <td>{_format_number(record.get('prefill_tokens_per_second'), suffix=' tok/s')}</td>
                <td>{_memory_label(record, 'allocated_bytes')}</td>
                <td>{_memory_label(record, 'reserved_bytes')}</td>
                <td>{_memory_label(record, 'device_free_bytes')}</td>
            </tr>
        ''')

    return f'''
        <div class="perf-table-wrap">
            <table class="perf-table">
                <thead><tr>
                    <th>Job</th><th>Status</th><th>Decode</th><th>End-to-end</th>
                    <th>Prefill</th><th>First output</th><th>Output</th><th>Prompt</th>
                    <th>Cached</th><th>Draft accepted</th>
                    <th>Chunk</th><th>Uncached prefill rate</th><th>Allocated at exit</th><th>Reserved at exit</th><th>Device free at exit</th>
                </tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
    '''


def _empty_state(title, message):
    return f'''
        <section class="perf-dashboard">
            <div class="perf-empty-state">
                <div class="perf-empty-icon">↗</div>
                <h2>{html.escape(title)}</h2>
                <p>{html.escape(message)}</p>
            </div>
        </section>
    '''


def render_performance_dashboard():
    model = shared.model
    if model is None:
        return _empty_state('No model loaded', 'Load a model, then complete a generation to see performance data.')
    if not hasattr(model, 'get_performance_stats'):
        return _empty_state('Metrics unavailable', 'The active loader does not expose detailed performance data.')

    try:
        performance = model.get_performance_stats() or {}
    except Exception as error:
        return _empty_state('Could not read metrics', str(error))

    records = list(performance.get('recent_requests') or [])
    model_name = html.escape(str(shared.model_name or 'Loaded model'))
    loader = html.escape(str(shared.args.loader or model.__class__.__name__))
    updated = datetime.now().astimezone().strftime('%H:%M:%S')

    if not records:
        return f'''
            <section class="perf-dashboard">
                <header class="perf-header">
                    <div><div class="perf-eyebrow">MODEL PERFORMANCE</div><h1>{model_name}</h1><p>{loader}</p></div>
                    <div class="perf-live"><i></i> Live · {REFRESH_SECONDS}s</div>
                </header>
                <div class="perf-empty-state perf-empty-compact">
                    <div class="perf-empty-icon">↗</div><h2>Waiting for a generation</h2>
                    <p>Metrics will appear here after the first request finishes.</p>
                </div>
            </section>
        '''

    latest = records[-1]
    displayed_records = records[-100:]
    completed = [record for record in records if record.get('completed')]
    latest_completed = completed[-1] if completed else latest
    cache = performance.get('image_cache') or {}
    cache_bytes = _number(cache.get('bytes'))
    cache_max = _number(cache.get('max_bytes'))
    cache_detail = f"{_format_integer(cache.get('entries'))} image entries"
    if cache_bytes is not None and cache_max:
        cache_detail += f' · {cache_bytes / 1024 / 1024:.1f}/{cache_max / 1024 / 1024:.0f} MiB'

    drafting = performance.get('drafting') or {}
    draft_detail = f"job #{latest_completed.get('job_id', '—')}"
    if drafting.get('mode') and drafting['mode'] != 'none':
        strategy = 'adaptive' if drafting.get('adaptive') else 'fixed'
        draft_detail = f"{drafting['mode']} · {strategy} · max {drafting.get('max_tokens', '—')}"
        if drafting.get('adaptive'):
            draft_detail += f" · target {_format_percent(drafting.get('confidence'))}"

    cards = ''.join([
        _metric_card('Decode speed', _format_number(_rate(latest_completed), suffix=' tok/s'), 'Backend token generation', 'blue'),
        _metric_card('End-to-end', _format_number(_rate(latest_completed, 'end_to_end'), suffix=' tok/s'), 'Includes request latency', 'green'),
        _metric_card('Time to first output', _format_number(latest_completed.get('time_to_first_output'), 3, 's'), 'Queue + prefill + first decode', 'orange'),
        _metric_card('Prompt cache', f"{_format_integer(latest_completed.get('cached_tokens'))} tokens", f"of {_format_integer(latest_completed.get('prompt_tokens'))} prompt tokens", 'violet'),
        _metric_card('Draft acceptance', _format_percent(latest_completed.get('draft_acceptance')), draft_detail, 'pink'),
        _metric_card('Image cache', _format_integer(cache.get('entries')), cache_detail, 'cyan'),
    ])

    return f'''
        <section class="perf-dashboard">
            <header class="perf-header">
                <div>
                    <div class="perf-eyebrow">MODEL PERFORMANCE</div>
                    <h1>{model_name}</h1>
                    <p>{loader} · {len(records)} recent request{'s' if len(records) != 1 else ''} · updated {updated}</p>
                </div>
                <div class="perf-live"><i></i> Live · {REFRESH_SECONDS}s</div>
            </header>
            <div class="perf-metrics">{cards}</div>
            <div class="perf-chart-grid">
                <article class="perf-panel">
                    <div class="perf-panel-heading"><h2>Throughput trend</h2><p>Completed requests</p></div>
                    {_line_chart(displayed_records)}
                </article>
                <article class="perf-panel">
                    <div class="perf-panel-heading"><h2>Latency by job</h2><p>Latest 16 completed requests</p></div>
                    {_latency_chart(records)}
                </article>
            </div>
            <article class="perf-panel perf-history">
                <div class="perf-panel-heading">
                    <h2>Request history</h2>
                    <p>Latest {len(displayed_records)} of {len(records)} retained requests · chunk size {_format_integer(performance.get('max_chunk_size'))}</p>
                </div>
                {_history_table(displayed_records)}
            </article>
        </section>
    '''


def create_ui():
    import gradio as gr

    with gr.Tab('Performance', elem_id='performance-tab'):
        shared.gradio['performance_dashboard'] = gr.HTML(
            value=render_performance_dashboard,
            every=REFRESH_SECONDS,
            elem_id='performance-dashboard',
            show_label=False,
        )
