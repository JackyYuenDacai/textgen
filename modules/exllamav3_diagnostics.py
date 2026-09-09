"""Request-boundary diagnostics; never synchronize, clear, or reset CUDA state."""
import time


def memory_snapshot(cuda):
    snapshot = {'sampled_at': time.time(), 'devices': []}
    if not cuda.is_initialized():
        return snapshot
    for device in range(cuda.device_count()):
        try:
            stats = cuda.memory_stats(device)
            if not stats:
                # Querying free memory can initialize a CUDA context on an
                # otherwise unused GPU. Do not spend VRAM merely to monitor it.
                snapshot['devices'].append({'device': device, 'allocator_uninitialized': True})
                continue
            free, total = cuda.mem_get_info(device)
            snapshot['devices'].append({
                'device': device,
                'allocated_bytes': stats.get('allocated_bytes.all.current'),
                'reserved_bytes': stats.get('reserved_bytes.all.current'),
                'inactive_split_bytes': stats.get('inactive_split_bytes.all.current'),
                # These peaks belong to the process, not this job. Do not reset
                # global counters: another request may be using the same GPU.
                'process_peak_allocated_bytes': stats.get('allocated_bytes.all.peak'),
                'process_peak_reserved_bytes': stats.get('reserved_bytes.all.peak'),
                'allocation_retries': stats.get('num_alloc_retries'),
                'ooms': stats.get('num_ooms'),
                'device_free_bytes': free,
                'device_total_bytes': total,
            })
        except Exception as error:
            snapshot['devices'].append({'device': device, 'error': str(error)})
    return snapshot


def recurrent_snapshot(generator):
    cache = getattr(generator, 'recurrent_cache', None)
    if cache is None:
        return None
    return {'entries': len(cache), 'bytes': cache.current_size, 'max_bytes': cache.max_size,
            **dict(cache.metrics)}


def recurrent_options(args):
    mib = args.exl3_recurrent_cache_mib
    interval = args.exl3_recurrent_checkpoint_interval
    prefill_interval = args.exl3_recurrent_checkpoint_interval_pp
    if not isinstance(mib, int) or mib < 1:
        raise ValueError('exl3-recurrent-cache-mib must be a positive integer.')
    for name, value in (('interval', interval), ('interval-pp', prefill_interval)):
        if name == 'interval' and value == 0:
            continue
        if not isinstance(value, int) or value < 256 or value % 256:
            raise ValueError(f'exl3-recurrent-checkpoint-{name} must be a positive multiple of 256.')
    return dict(recurrent_cache_size=mib * 1024**2,
                recurrent_checkpoint_interval=interval or None,
                recurrent_checkpoint_interval_pp=prefill_interval)
