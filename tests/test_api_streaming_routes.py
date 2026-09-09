import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch('sys.argv', ['textgen-tests']):
    from modules.api import script
    from modules.api.typing import ChatCompletionRequest, CompletionRequest

from modules.api.streaming_response import GenerationEventSourceResponse


class StreamingRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.exit_patch = patch('sse_starlette.sse.AppStatus.should_exit_event', None)
        self.exit_patch.start()
        self.addCleanup(self.exit_patch.stop)

    async def check_route(self, kind):
        entered, finished = threading.Event(), threading.Event()
        observed_events = []
        timed_out = []
        def fake_stream(*args, stop_event, **kwargs):
            observed_events.append(stop_event)
            entered.set()
            try:
                if not stop_event.wait(timeout=3):
                    timed_out.append(True)
                    yield {}
            finally:
                finished.set()
        request = SimpleNamespace(url=SimpleNamespace(path='/v1/' + kind),
                                  is_disconnected=Mock(side_effect=AssertionError('Competing receive consumer')))
        function = 'stream_completions' if kind == 'completions' else 'stream_chat_completions'
        with patch.object(script.OAIcompletions, function, side_effect=fake_stream):
            if kind == 'completions':
                result = await script.openai_completions(request, CompletionRequest(prompt='test', stream=True))
            elif kind == 'chat/completions':
                result = await script.openai_chat_completions(request, ChatCompletionRequest(
                    messages=[{'role': 'user', 'content': 'test'}], stream=True))
            else:
                result = await script._anthropic_generate(request, SimpleNamespace(stream=True), {}, 'test-model')
            self.assertIsInstance(result, GenerationEventSourceResponse)
            async def receive():
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                return {'type': 'http.disconnect'}
            async def send(message):
                pass
            try:
                await asyncio.wait_for(result({'type': 'http'}, receive, send), timeout=2)
                self.assertEqual(observed_events, [result.stop_event])
                self.assertTrue(result.stop_event.is_set())
                self.assertTrue(finished.is_set())
                self.assertEqual(timed_out, [])
                request.is_disconnected.assert_not_called()
            finally:
                result.stop_event.set()
                await result.body_iterator.aclose()

    async def test_completions_disconnect_during_prefill(self):
        await self.check_route('completions')

    async def test_chat_disconnect_during_prefill(self):
        await self.check_route('chat/completions')

    async def test_anthropic_disconnect_during_prefill(self):
        await self.check_route('messages')


if __name__ == '__main__':
    unittest.main()
