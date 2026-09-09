"""Reclaim unused allocator blocks under device pressure, between inference rounds.

WDDM may page CUDA allocations instead of raising OOM, so allocator OOM retries
alone do not protect decode latency. This does not evict KV, weights or graphs.
"""
import time


class MemoryPressureGuard:
    def __init__(self, cuda, clock=time.monotonic, free_mib=1024, reclaim_mib=256):
        self.cuda = cuda
        self.clock = clock
        self.free_bytes = free_mib * 1024**2
        self.reclaim_bytes = reclaim_mib * 1024**2
        self.next_check = 0.0
        self.next_reclaim = 0.0
        self.reclaims = 0
        self.last_event = None

    def check(self):
        now = self.clock()
        if now < self.next_check or now < self.next_reclaim:
            return None
        self.next_check = now + 1.0
        if not self.cuda.is_initialized():
            return None
        try:
            candidates = []
            for device in range(self.cuda.device_count()):
                stats = self.cuda.memory_stats(device)
                if not stats:
                    continue  # Do not initialize an unused GPU for monitoring.
                reserved = stats.get('reserved_bytes.all.current', 0)
                allocated = stats.get('allocated_bytes.all.current', 0)
                split = stats.get('inactive_split_bytes.all.current', 0)
                # Inactive split fragments generally cannot be released to CUDA.
                releasable = max(0, reserved - allocated - split)
                free, _ = self.cuda.mem_get_info(device)
                if free < self.free_bytes and releasable >= self.reclaim_bytes:
                    candidates.append((device, reserved, free, releasable))
            if not candidates:
                return None
            # Set cooldown even if reclamation fails. Never clear every token.
            self.next_reclaim = now + 5.0
            self.cuda.empty_cache()
            self.reclaims += 1
            devices = []
            for device, reserved, free, releasable in candidates:
                after = self.cuda.memory_reserved(device)
                free_after, _ = self.cuda.mem_get_info(device)
                devices.append(dict(device=device, reserved_before=reserved, reserved_after=after,
                                    released_bytes=max(0, reserved - after),
                                    free_before=free, free_after=free_after,
                                    estimated_releasable_bytes=releasable))
            self.last_event = dict(reclaims=self.reclaims, devices=devices)
            return self.last_event
        except Exception as error:
            self.next_reclaim = now + 5.0
            self.last_event = dict(reclaims=self.reclaims, error=str(error))
            return self.last_event

    def info(self):
        return dict(reclaims=self.reclaims, last_event=self.last_event,
                    low_free_threshold_bytes=self.free_bytes)
