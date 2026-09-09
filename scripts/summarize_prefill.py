"""Compare saved performance snapshots without contacting or reloading a server."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize(paths):
    groups = defaultdict(list)
    seen = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        performance = data.get('performance', data)
        session = performance.get('history', {}).get('session_id')
        for record in performance.get('recent_requests', []):
            if session and record.get('sequence') is not None:
                identity = (data.get('base'), session, record['sequence'])
                if identity in seen:
                    continue
                seen.add(identity)
            config = record.get('configuration') or {}
            # Older records cannot be attributed to a chunk setting reliably.
            if not record.get('completed') or not config or not record.get('time_prefill', 0) > 0:
                continue
            prompt, cached = record.get('prompt_tokens'), record.get('cached_tokens')
            if prompt is None or cached is None or prompt <= cached:
                continue
            uncached = prompt - cached
            key = (data.get('model_name'), config.get('context_capacity'), config.get('cache_type'),
                   config.get('max_chunk_size'), config.get('recurrent_checkpoint_interval'),
                   config.get('recurrent_checkpoint_interval_pp'),
                   config.get('recurrent_cache_mib'), config.get('staging_mode'),
                   config.get('staging_bucket_pages'), json.dumps(config.get('drafting'), sort_keys=True),
                   'cold' if cached == 0 else 'warm', prompt // 16384 * 16384,
                   uncached // 4096 * 4096)
            groups[key].append(record)
    result = []
    keys = ('model', 'context_capacity', 'cache_type', 'chunk', 'tail_checkpoint',
            'prefill_checkpoint', 'checkpoint_ram_mib', 'staging_mode', 'staging_bucket_pages',
            'drafting', 'cache_state', 'prompt_band_start', 'uncached_band_start')
    for group, records in groups.items():
        tokens = sum(r['prompt_tokens'] - r['cached_tokens'] for r in records)
        seconds = sum(r['time_prefill'] for r in records)
        free_samples = [d['device_free_bytes'] for r in records
                        for phase in ('memory_before', 'memory_first_output', 'memory_after')
                        for d in (r.get(phase) or {}).get('devices', []) if 'device_free_bytes' in d]
        result.append(dict(zip(keys, group), jobs=len(records), uncached_tokens=tokens,
                           prefill_seconds=seconds, prefill_tokens_per_second=tokens / seconds,
                           minimum_sampled_device_free_mib=min(free_samples) / 1024**2 if free_samples else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='+')
    parser.add_argument('--out', required=True, help='Output CSV')
    args = parser.parse_args()
    rows = summarize(args.snapshots)
    if not rows:
        parser.error('No completed records with per-request configuration; collect snapshots after restarting with the new diagnostics.')
    with open(args.out, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} comparison groups. Rates are token-weighted; sampled free memory is not a request peak.')


if __name__ == '__main__':
    main()
