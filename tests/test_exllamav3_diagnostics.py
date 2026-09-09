import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from modules.exllamav3_diagnostics import memory_snapshot, recurrent_options, recurrent_snapshot
from scripts.summarize_prefill import summarize

# Import the pure geometry helper without creating a CUDA context or loading kernels.
spec = importlib.util.spec_from_file_location('staging_geometry',
    Path(__file__).resolve().parents[1] / 'exllamav3/exllamav3/util/staging.py')
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


class StagingTests(unittest.TestCase):
    def test_boundary_savings_and_full_coverage(self):
        self.assertEqual(staging.staging_pages(513), 576)
        self.assertEqual(staging.staging_pages(513, 0), 1024)
        # 256 tokens/page * 4 KV heads * 256 head dim * 2 bytes * K/V.
        self.assertEqual((1024 - 576) * 256 * 4 * 256 * 2 * 2, 448 * 1024**2)
        for batch in (1, 2, 7):
            for pages in (1, 63, 64, 65, 511, 512, 513, 824, 938, 1025):
                required = batch * pages
                allocated = staging.staging_pages(required)
                self.assertGreaterEqual(allocated, required)
                self.assertLess(allocated - required, 64)
                # Every remapped logical page index fits the staging buffer.
                self.assertLess(batch * pages - 1, allocated)

    def test_small_prompts_and_invalid_values(self):
        self.assertEqual(staging.staging_pages(1), 1)
        self.assertEqual(staging.staging_pages(8), 8)
        for args in ((-1, 64), (10, -1)):
            with self.assertRaises(ValueError):
                staging.staging_pages(*args)


class DiagnosticsTests(unittest.TestCase):
    def test_snapshot_never_initializes_or_resets_cuda(self):
        cuda = Mock()
        cuda.is_initialized.return_value = False
        self.assertEqual(memory_snapshot(cuda)['devices'], [])
        cuda.device_count.assert_not_called()
        cuda.empty_cache.assert_not_called()
        cuda.reset_peak_memory_stats.assert_not_called()

    def test_multiple_devices_and_partial_failure_are_serializable(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.device_count.return_value = 2
        cuda.memory_stats.side_effect = [{'allocated_bytes.all.current': 100,
                                         'reserved_bytes.all.current': 200,
                                         'allocated_bytes.all.peak': 300}, RuntimeError('unavailable')]
        cuda.mem_get_info.return_value = (400, 1000)
        snapshot = memory_snapshot(cuda)
        first, second = snapshot['devices']
        self.assertEqual(first['allocated_bytes'], 100)
        self.assertEqual(first['process_peak_allocated_bytes'], 300)
        self.assertIsNone(first['inactive_split_bytes'])
        self.assertIn('error', second)
        json.dumps(snapshot)
        cuda.synchronize.assert_not_called()
        cuda.reset_peak_memory_stats.assert_not_called()

    def test_unused_device_does_not_get_a_context_for_monitoring(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.device_count.return_value = 1
        cuda.memory_stats.return_value = {}
        self.assertTrue(memory_snapshot(cuda)['devices'][0]['allocator_uninitialized'])
        cuda.mem_get_info.assert_not_called()

    def test_checkpoint_options_and_empty_cache(self):
        args = SimpleNamespace(exl3_recurrent_cache_mib=4096,
                               exl3_recurrent_checkpoint_interval=2048,
                               exl3_recurrent_checkpoint_interval_pp=8192)
        self.assertEqual(recurrent_options(args)['recurrent_checkpoint_interval_pp'], 8192)
        for name, value in (('exl3_recurrent_cache_mib', 0),
                            ('exl3_recurrent_checkpoint_interval_pp', 257),
                            ('exl3_recurrent_checkpoint_interval', 257)):
            bad = SimpleNamespace(**vars(args))
            setattr(bad, name, value)
            with self.assertRaises(ValueError):
                recurrent_options(bad)
        class EmptyCache(dict):
            current_size = 0
            max_size = 4096
            metrics = {'stash_evictions': 0}
        self.assertEqual(recurrent_snapshot(SimpleNamespace(recurrent_cache=EmptyCache()))['entries'], 0)
        args.exl3_recurrent_checkpoint_interval = 0
        self.assertIsNone(recurrent_options(args)['recurrent_checkpoint_interval'])

    def test_offline_comparison_deduplicates_and_separates_workloads(self):
        config = dict(max_chunk_size=4096, context_capacity=240128, cache_type='q4')
        first = dict(sequence=1, completed=True, configuration=config,
                     prompt_tokens=160100, cached_tokens=160000, time_prefill=1)
        second = dict(first, sequence=2, prompt_tokens=160300, time_prefill=2)
        cold = dict(first, sequence=3, cached_tokens=0, time_prefill=100)
        data = dict(model_name='model', base='local', performance=dict(
            history=dict(session_id='session'), recent_requests=[first, second, cold]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            rows = summarize([path, path])
        self.assertEqual(len(rows), 2)
        warm = next(row for row in rows if row['cache_state'] == 'warm')
        self.assertEqual(warm['jobs'], 2)
        self.assertAlmostEqual(warm['prefill_tokens_per_second'], 400 / 3)
        self.assertIsNone(warm['minimum_sampled_device_free_mib'])


if __name__ == '__main__':
    unittest.main()
