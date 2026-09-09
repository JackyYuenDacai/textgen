import unittest
from unittest.mock import Mock

from modules.exllamav3_memory import MemoryPressureGuard

MIB = 1024**2


class MemoryPressureTests(unittest.TestCase):
    def make(self, free=100, allocated=17000, reserved=21000, split=900):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.device_count.return_value = 1
        cuda.memory_stats.return_value = {
            'allocated_bytes.all.current': allocated * MIB,
            'reserved_bytes.all.current': reserved * MIB,
            'inactive_split_bytes.all.current': split * MIB,
        }
        cuda.mem_get_info.return_value = (free * MIB, 24000 * MIB)
        cuda.memory_reserved.return_value = (allocated + split) * MIB
        clock = Mock(return_value=10.0)
        return cuda, clock, MemoryPressureGuard(cuda, clock)

    def test_low_free_reclaims_whole_unused_blocks_without_touching_live_cache(self):
        cuda, clock, guard = self.make()
        event = guard.check()
        self.assertEqual(event['devices'][0]['released_bytes'], 3100 * MIB)
        cuda.empty_cache.assert_called_once_with()
        cuda.reset_peak_memory_stats.assert_not_called()
        cuda.set_per_process_memory_fraction.assert_not_called()
        self.assertEqual(guard.info()['reclaims'], 1)

    def test_healthy_headroom_keeps_allocator_reuse(self):
        cuda, _, guard = self.make(free=1500)
        self.assertIsNone(guard.check())
        cuda.empty_cache.assert_not_called()

    def test_live_memory_or_fragmentation_alone_does_not_cause_flush_loop(self):
        cuda, _, guard = self.make(reserved=18000, split=950)
        self.assertIsNone(guard.check())
        cuda.empty_cache.assert_not_called()

    def test_checks_and_reclaims_are_rate_limited(self):
        cuda, clock, guard = self.make()
        guard.check()
        for now in (10.01, 10.5, 11, 14.9):
            clock.return_value = now
            self.assertIsNone(guard.check())
        cuda.empty_cache.assert_called_once()
        clock.return_value = 15.0
        guard.check()
        self.assertEqual(cuda.empty_cache.call_count, 2)

    def test_unused_device_and_uninitialized_cuda_are_not_touched(self):
        cuda, _, guard = self.make()
        cuda.is_initialized.return_value = False
        guard.check()
        cuda.memory_stats.assert_not_called()
        cuda, _, guard = self.make()
        cuda.memory_stats.return_value = {}
        guard.check()
        cuda.mem_get_info.assert_not_called()

    def test_failure_is_reported_without_aborting_generation_or_retry_storm(self):
        cuda, clock, guard = self.make()
        cuda.empty_cache.side_effect = RuntimeError('unavailable')
        self.assertIn('error', guard.check())
        clock.return_value = 11.0
        self.assertIsNone(guard.check())
        cuda.empty_cache.assert_called_once()


if __name__ == '__main__':
    unittest.main()
