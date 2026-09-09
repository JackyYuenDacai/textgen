import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch('sys.argv', ['textgen-tests']):
    from modules import ui_performance


class PerformanceDashboardTests(unittest.TestCase):
    def test_memory_labels_handle_old_records_and_escape_device_names(self):
        self.assertEqual(ui_performance._memory_label({}, 'allocated_bytes'), 'N/A')
        record = {'memory_after': {'devices': [
            {'device': '<test>', 'allocated_bytes': 2 * 1024**2},
            {'device': 1, 'error': 'unavailable'}]}}
        self.assertEqual(ui_performance._memory_label(record, 'allocated_bytes'), 'GPU &lt;test&gt;: 2 MiB')

    def test_large_history_keeps_dashboard_bounded_without_hiding_retained_count(self):
        records = [{'job_id': job_id, 'completed': True, 'new_tokens': 4,
                    'time_generate': 1, 'time_prefill': 0.5} for job_id in range(120)]
        model = SimpleNamespace(get_performance_stats=lambda: {'recent_requests': records})
        with patch.object(ui_performance.shared, 'model', model), \
             patch.object(ui_performance.shared, 'model_name', 'test-model'), \
             patch.object(ui_performance.shared.args, 'loader', 'ExLlamav3'):
            output = ui_performance.render_performance_dashboard()
        self.assertIn('Latest 100 of 120 retained requests', output)
        self.assertEqual(len(records), 120)

    def test_populated_dashboard_renders_metrics_charts_and_escaped_model_name(self):
        stats = {
            'recent_requests': [
                {
                    'job_id': 7,
                    'completed': True,
                    'new_tokens': 100,
                    'emitted_tokens': 99,
                    'total_seconds': 2,
                    'decode_tokens_per_second': 80,
                    'time_prefill': 0.5,
                    'time_generate': 1.25,
                    'time_to_first_output': 0.6,
                    'prompt_tokens': 1000,
                    'cached_tokens': 768,
                    'draft_acceptance': 0.75,
                },
                {
                    'job_id': 8,
                    'completed': False,
                    'emitted_tokens': 4,
                    'total_seconds': 0.2,
                },
            ],
            'image_cache': {'entries': 2, 'bytes': 1048576, 'max_bytes': 4194304},
            'max_chunk_size': 2048,
            'drafting': {'mode': 'mtp', 'adaptive': True, 'max_tokens': 5, 'confidence': 0.4},
        }
        model = SimpleNamespace(get_performance_stats=lambda: stats)
        with patch.object(ui_performance.shared, 'model', model), \
             patch.object(ui_performance.shared, 'model_name', '<Qwen & friends>'), \
             patch.object(ui_performance.shared.args, 'loader', 'ExLlamav3'):
            output = ui_performance.render_performance_dashboard()

        self.assertIn('&lt;Qwen &amp; friends&gt;', output)
        self.assertIn('80.00 tok/s', output)
        self.assertIn('50.00 tok/s', output)
        self.assertIn('75.0%', output)
        self.assertIn('mtp · adaptive · max 5 · target 40.0%', output)
        self.assertIn('Throughput trend', output)
        self.assertIn('Latency by job', output)
        self.assertIn('perf-status-stopped', output)
        self.assertNotIn('<Qwen & friends>', output)

    def test_empty_and_unsupported_states_are_clear(self):
        with patch.object(ui_performance.shared, 'model', None):
            self.assertIn('No model loaded', ui_performance.render_performance_dashboard())

        with patch.object(ui_performance.shared, 'model', object()):
            self.assertIn('Metrics unavailable', ui_performance.render_performance_dashboard())

    def test_invalid_values_do_not_break_rendering(self):
        stats = {
            'recent_requests': [{
                'job_id': 'bad<script>',
                'completed': True,
                'new_tokens': None,
                'total_seconds': 0,
                'decode_tokens_per_second': float('nan'),
                'time_prefill': 'invalid',
            }]
        }
        model = SimpleNamespace(get_performance_stats=lambda: stats)
        with patch.object(ui_performance.shared, 'model', model), \
             patch.object(ui_performance.shared, 'model_name', 'model'), \
             patch.object(ui_performance.shared.args, 'loader', None):
            output = ui_performance.render_performance_dashboard()

        self.assertIn('bad&lt;script&gt;', output)
        self.assertNotIn('nan', output.lower())
        self.assertIn('—', output)


if __name__ == '__main__':
    unittest.main()
