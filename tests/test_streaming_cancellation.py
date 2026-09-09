import asyncio
import threading
import unittest
from unittest.mock import patch

from starlette.concurrency import iterate_in_threadpool

from modules.api.streaming_response import GenerationEventSourceResponse


class StreamingCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.exit_patch = patch('sse_starlette.sse.AppStatus.should_exit_event', None)
        self.exit_patch.start()
        self.addCleanup(self.exit_patch.stop)

    async def test_disconnect_signals_stop_before_the_first_output(self):
        stop_event = threading.Event()
        entered = threading.Event()
        finished = threading.Event()
        timed_out = []
        sent = []
        def response():
            entered.set()
            try:
                if not stop_event.wait(timeout=2):
                    timed_out.append(True)
                    yield {'data': 'late token'}
            finally:
                finished.set()
        async def stream():
            chunks = response()
            try:
                async for chunk in iterate_in_threadpool(chunks):
                    yield chunk
            finally:
                chunks.close()
        async def receive():
            while not entered.is_set():
                await asyncio.sleep(0.001)
            return {'type': 'http.disconnect'}
        async def send(message):
            sent.append(message)
        result = GenerationEventSourceResponse(stream(), stop_event, sep='\n')
        try:
            await asyncio.wait_for(result({'type': 'http'}, receive, send), timeout=1)
            self.assertTrue(stop_event.is_set())
            self.assertTrue(finished.is_set())
            self.assertEqual(timed_out, [])
            self.assertFalse(any(b'late token' in message.get('body', b'') for message in sent))
        finally:
            stop_event.set()

    async def test_normal_sse_output_and_completion_are_preserved(self):
        stop_event = threading.Event()
        sent = []
        async def stream():
            yield {'data': 'hello'}
            yield {'data': '[DONE]'}
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            sent.append(message)
        result = GenerationEventSourceResponse(stream(), stop_event, sep='\n')
        await asyncio.wait_for(result({'type': 'http'}, receive, send), timeout=1)
        body = b''.join(message.get('body', b'') for message in sent)
        self.assertIn(b'data: hello\n\n', body)
        self.assertIn(b'data: [DONE]\n\n', body)
        self.assertFalse(sent[-1]['more_body'])
        self.assertTrue(stop_event.is_set())

    async def test_request_body_messages_do_not_cancel_and_events_are_request_local(self):
        first_stop, second_stop = threading.Event(), threading.Event()
        async def content():
            yield {'data': 'unused'}
        first = GenerationEventSourceResponse(content(), first_stop)
        second = GenerationEventSourceResponse(content(), second_stop)
        messages = iter([{'type': 'http.request', 'body': b'', 'more_body': False}, {'type': 'http.disconnect'}])
        async def receive():
            return next(messages)
        async def send(message):
            pass
        async def drive_response(scope, wrapped_receive, sender):
            self.assertEqual((await wrapped_receive())['type'], 'http.request')
            self.assertFalse(first_stop.is_set())
            self.assertEqual((await wrapped_receive())['type'], 'http.disconnect')
            self.assertTrue(first_stop.is_set())
        with patch('sse_starlette.EventSourceResponse.__call__', side_effect=drive_response):
            await first({'type': 'http'}, receive, send)
        self.assertFalse(second_stop.is_set())
        await first.body_iterator.aclose()
        await second.body_iterator.aclose()

    async def test_send_failure_sets_stop(self):
        stop_event = threading.Event()
        async def content():
            yield {'data': 'hello'}
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            raise OSError('connection closed')
        result = GenerationEventSourceResponse(content(), stop_event)
        with self.assertRaises(Exception):
            await result({'type': 'http'}, receive, send)
        self.assertTrue(stop_event.is_set())
        await result.body_iterator.aclose()


if __name__ == '__main__':
    unittest.main()
