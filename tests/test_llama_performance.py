import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import requests

with patch('sys.argv', ['textgen-tests']):
    from modules import shared, ui_performance
    from modules.llama_cpp_server import LlamaServer


class LlamaPerformanceTests(unittest.TestCase):
    def setUp(self):
        for name, value in [('is_multimodal', False), ('stop_everything', False)]:
            context = patch.object(shared, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(shared.args, 'verbose', False)
        context.start()
        self.addCleanup(context.stop)
        self.server = self.make_server()
        self.state = {'add_bos_token': False, 'auto_max_new_tokens': False, 'max_new_tokens': 256}

    def make_server(self):
        with patch.object(LlamaServer, '_start_server'), patch.object(LlamaServer, '_find_available_port', return_value=5005):
            server = LlamaServer('Ternary-Bonsai-2-27B-PQ2_0.gguf')
        server.session.close()
        server.session = Mock()
        server.prepare_payload = Mock(return_value={})
        server.encode = Mock(return_value=[1, 2, 3])
        return server

    def stream(self, events, status=200):
        response = Mock(status_code=status)
        response.iter_lines.return_value = [
            b'data: ' + json.dumps(event).encode() if isinstance(event, dict) else event
            for event in events
        ]
        self.server.session.post.return_value = response
        return response

    def final_event(self):
        return {'stop': True, 'stop_type': 'limit', 'tokens_predicted': 256,
                'tokens_evaluated': 55000, 'tokens_cached': 55255,
                'timings': {'cache_n': 1000, 'prompt_n': 54000,
                            'prompt_ms': 20000, 'prompt_per_second': 2700,
                            'predicted_n': 256, 'predicted_ms': 3221.713,
                            'predicted_per_second': 79.1504,
                            'draft_n': 100, 'draft_n_accepted': 75}}

    def record(self):
        return self.server.get_performance_stats()['recent_requests'][-1]

    def test_native_timings_reach_dashboard_with_authoritative_counts(self):
        response = self.stream([{'content': 'Many tokens', 'tokens_predicted': 9}, self.final_event()])
        with patch('modules.llama_cpp_server.time.perf_counter', side_effect=[10, 10.5, 34]):
            self.assertEqual(self.server.generate('prompt', self.state), 'Many tokens')
        record = self.record()
        self.assertTrue(record['completed'])
        self.assertEqual(record['new_tokens'], 256)
        self.assertEqual(self.server.last_completion_token_count, 256)
        self.assertEqual(record['prompt_tokens'], 55000)
        self.assertEqual(record['cached_tokens'], 1000)  # Not the KV size at exit.
        self.assertEqual(record['decode_tokens_per_second'], 79.1504)
        self.assertAlmostEqual(record['time_generate'], 3.221713)
        self.assertEqual(record['time_prefill'], 20)
        self.assertEqual(record['prefill_tokens_per_second'], 2700)
        self.assertEqual(record['time_to_first_output'], 0.5)
        self.assertEqual(record['total_seconds'], 24)
        self.assertEqual(record['draft_acceptance'], 0.75)
        self.assertEqual(record['eos_reason'], 'limit')
        response.close.assert_called_once()
        with patch.object(shared, 'model', self.server), patch.object(shared.args, 'loader', 'llama.cpp'):
            dashboard = ui_performance.render_performance_dashboard()
        self.assertIn('79.15 tok/s', dashboard)
        self.assertIn('10.67 tok/s', dashboard)
        self.assertIn('75.0%', dashboard)
        self.assertIn('55,000', dashboard)
        self.assertNotIn('Metrics unavailable', dashboard)

    def test_final_chunk_is_captured_before_consumer_closes(self):
        final = self.final_event()
        final.update(content='last', completion_probabilities=[{'id': 7}])
        response = self.stream([final])
        generator = self.server.generate_with_streaming('prompt', self.state)
        self.assertEqual(next(generator), 'last')
        generator.close()
        self.assertTrue(self.record()['completed'])
        self.assertEqual(self.record()['new_tokens'], 256)
        self.assertEqual(self.server.last_completion_probabilities, [{'id': 7}])
        response.close.assert_called_once()

    def test_cancellation_keeps_known_count_without_inventing_decode_speed(self):
        self.stream([{'content': 'a speculative chunk', 'tokens_predicted': 12}, self.final_event()])
        generator = self.server.generate_with_streaming('prompt', self.state)
        next(generator)
        generator.close()
        self.assertFalse(self.record()['completed'])
        self.assertEqual(self.record()['new_tokens'], 12)
        self.assertIsNone(self.record()['decode_tokens_per_second'])
        self.assertNotIn('time_generate', self.record())

    def test_stop_event_and_global_stop_are_incomplete(self):
        for global_stop in (True, False):
            with self.subTest(global_stop=global_stop):
                stop_event = threading.Event()
                stop_event.set()
                self.state['stop_event'] = None if global_stop else stop_event
                self.stream([self.final_event()])
                with patch.object(shared, 'stop_everything', global_stop):
                    self.assertEqual(self.server.generate('prompt', self.state), '')
                self.assertFalse(self.record()['completed'])
                self.assertIsNone(self.record()['new_tokens'])

    def test_missing_timings_and_truncated_stream_do_not_reuse_previous_metrics(self):
        self.stream([self.final_event()])
        self.server.generate('prompt', self.state)
        for ending in ({'stop': True}, b'data: [DONE]', None):
            with self.subTest(ending=ending):
                events = [{'content': 'text'}]
                if ending is not None:
                    events.append(ending)
                self.stream(events)
                self.server.generate('prompt', self.state)
                self.assertIsNone(self.record()['new_tokens'])  # Chunks are not tokens.
                self.assertIsNone(self.record()['decode_tokens_per_second'])
                self.assertEqual(self.record()['completed'], isinstance(ending, dict))

    def test_http_and_connection_failures_record_incomplete_request(self):
        response = self.stream([])
        response.raise_for_status.side_effect = requests.HTTPError('failed')
        with self.assertRaises(requests.HTTPError):
            self.server.generate('prompt', self.state)
        response.close.assert_called_once()
        self.assertFalse(self.record()['completed'])
        self.server.session.post.side_effect = requests.ConnectionError('unavailable')
        with self.assertRaises(requests.ConnectionError):
            self.server.generate('prompt', self.state)
        self.assertEqual(len(self.server.get_performance_stats()['recent_requests']), 2)
        self.assertFalse(self.record()['completed'])

    def test_context_rejection_and_stream_failure_close_response(self):
        response = self.stream([], status=400)
        response.json.return_value = {'error': {'type': 'exceed_context_size_error'}}
        self.assertEqual(self.server.generate('prompt', self.state), '')
        response.close.assert_called_once()
        self.assertFalse(self.record()['completed'])
        response = self.stream([])
        response.iter_lines.side_effect = requests.ConnectionError('stream interrupted')
        with self.assertRaises(requests.ConnectionError):
            self.server.generate('prompt', self.state)
        response.close.assert_called_once()
        self.assertFalse(self.record()['completed'])

    def test_older_timing_schema_and_missing_token_count(self):
        event = self.final_event()
        del event['timings']['cache_n']
        del event['tokens_predicted']
        self.stream([event])
        self.server.generate('prompt', self.state)
        self.assertEqual(self.record()['cached_tokens'], 1000)
        self.assertEqual(self.record()['new_tokens'], 256)

    def test_multimodal_actual_prompt_count_replaces_estimate(self):
        self.server._process_images_for_generation = Mock(return_value=[object()])
        self.stream([self.final_event()])
        with patch.object(shared, 'is_multimodal', True), patch('modules.llama_cpp_server.convert_pil_to_base64', return_value='image'):
            self.server.generate('<__media__>prompt', self.state)
        self.assertEqual(self.record()['prompt_tokens'], 55000)
        self.assertEqual(self.server.session.post.call_args.kwargs['json']['prompt']['multimodal_data'], ['image'])

    def test_history_is_bounded_thread_safe_and_model_scoped(self):
        def record(_):
            self.server._record_performance({'prompt_tokens': 3, 'completed': True}, self.final_event())
            return self.server.get_performance_stats()
        with ThreadPoolExecutor(max_workers=4) as pool:
            snapshots = list(pool.map(record, range(125)))
        stats = self.server.get_performance_stats()
        self.assertEqual(stats['history']['total_recorded'], 125)
        self.assertEqual(stats['history']['dropped'], 25)
        self.assertEqual(stats['history']['first_sequence'], 26)
        self.assertEqual(stats['history']['last_sequence'], 125)
        self.assertEqual(len(stats['recent_requests']), 100)
        self.assertEqual([r['sequence'] for r in stats['recent_requests']], list(range(26, 126)))
        for snapshot in snapshots:
            self.assertEqual(snapshot['history']['retained'], len(snapshot['recent_requests']))
        stats['recent_requests'][-1]['new_tokens'] = 999
        self.assertEqual(self.record()['new_tokens'], 256)
        second = self.make_server().get_performance_stats()
        self.assertEqual(second['recent_requests'], [])
        self.assertNotEqual(second['history']['session_id'], stats['history']['session_id'])


if __name__ == '__main__':
    unittest.main()
