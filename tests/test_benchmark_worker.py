"""Run with the isolated benchmark interpreter; exercises the actual Inspect engine."""

import asyncio
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from scripts import capability_benchmark_worker as worker


@unittest.skipUnless(importlib.util.find_spec('inspect_ai'), 'Requires the optional benchmark environment')
class WorkerTests(unittest.TestCase):
    def test_real_inspect_scoring_through_chat_completions(self):
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_evals.gsm8k import gsm8k

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path, request))
                body = json.dumps({
                    'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'test-model',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ANSWER: 4'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            dataset = MemoryDataset([Sample(id='test-1', input='What is 2 + 2?', target='4')])
            with patch('inspect_evals.gsm8k.gsm8k.hf_dataset', return_value=dataset):
                task = gsm8k(fewshot=0)
            config = dict(task='gsm8k', endpoint=f'http://127.0.0.1:{server.server_port}/v1', model='test-model',
                          limit=1, seed=42, max_tokens=128, temperature=0, thinking='Disabled')
            with tempfile.TemporaryDirectory() as directory, patch.object(worker, 'make_task', return_value=task), \
                    patch.object(worker, 'emit') as emit, patch.dict(os.environ, {'TEXTGEN_BENCH_API_KEY': 'test-key'}):
                asyncio.run(worker.evaluate(Path(directory), config))
                events = [call.kwargs for call in emit.call_args_list]
                final = events[-1]
                self.assertEqual(final['status'], 'completed', events)
                self.assertTrue(any(row['metric'] == 'accuracy' and row['value'] == 1 for row in final['metrics']), final)
                sample = next(e['sample'] for e in events if e.get('kind') == 'sample')
                self.assertEqual(sample['answer'], 'ANSWER: 4')
                self.assertFalse(sample['error'])
                self.assertFalse(sample['truncated'])
                logs = list((Path(directory) / 'logs').glob('*.json'))
                self.assertTrue(logs)
                self.assertNotIn('test-key', logs[0].read_text(encoding='utf-8'))
            self.assertEqual(requests[0][0], '/v1/chat/completions')
            self.assertEqual(requests[0][1]['model'], 'test-model')
            self.assertEqual(requests[0][1]['max_tokens'], 128)
            self.assertFalse(requests[0][1]['enable_thinking'])
        finally:
            server.shutdown()
            server.server_close()

    def test_humaneval_does_not_fallback_to_host_execution(self):
        from inspect_evals.humaneval import humaneval  # noqa: F401

        with patch.object(worker.subprocess, 'run') as run:
            run.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'Docker'):
                worker.make_task({'task': 'humaneval'})

    def test_stop_cancels_inspect_and_reports_cancelled(self):
        from inspect_ai import Task
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import match
        from inspect_ai.solver import solver

        @solver
        def slow_solver():
            async def solve(state, generate):
                await asyncio.sleep(30)
                return state
            return solve

        task = Task(dataset=[Sample(id='cancel', input='Test', target='4')], solver=slow_solver(), scorer=match())
        config = dict(task='gsm8k', endpoint='http://127.0.0.1:1/v1', model='cancel-test',
                      limit=1, seed=42, max_tokens=128, temperature=0, thinking='Server default')
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, 'make_task', return_value=task), \
                patch.object(worker, 'emit') as emit, patch.dict(os.environ, {'TEXTGEN_BENCH_API_KEY': 'test-key'}):
            async def run():
                async def request_stop():
                    await asyncio.sleep(0.75)
                    (Path(directory) / 'stop').touch()
                stopper = asyncio.create_task(request_stop())
                await asyncio.wait_for(worker.evaluate(Path(directory), config), timeout=10)
                await stopper
            asyncio.run(run())
            self.assertEqual(emit.call_args_list[-1].kwargs['status'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
