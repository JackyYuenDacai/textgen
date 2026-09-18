"""Responses wire contracts and cancellation regressions; no model required."""
import asyncio
import json
import threading
import unittest
from unittest.mock import Mock, patch

from test_responses_api import api, script, decoded
from modules.api.generation_events import EventBatch, ReasoningDelta, TextDelta, ToolCallDelta, DoneEvent


FORMAT = {'format': {'type': 'json_schema', 'name': 'answer', 'strict': True,
                     'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                                'required': ['ok'], 'additionalProperties': False}}}


class WireTests(unittest.TestCase):
    def test_buffered_text_opens_with_empty_content_before_its_delta(self):
        request = api.ResponsesRequest(input='Hi', store=False, tools=[{'type': 'function', 'name': 'lookup'}])
        converter = api.StreamConverter(request, [], 'local')
        events = converter.start() + converter.process({'choices': [
            {'delta': {'content': 'Hello'}, 'finish_reason': 'stop'}]}) + converter.finish()
        data = decoded(events)
        part = next(e['part'] for e in data if e['type'] == 'response.content_part.added')
        self.assertEqual(part['text'], '')
        self.assertEqual(''.join(e['delta'] for e in data if e['type'] == 'response.output_text.delta'), 'Hello')
        self.assertEqual([e['sequence_number'] for e in data], list(range(len(data))))

    def test_mixed_native_events_have_contiguous_wire_sequence(self):
        request = api.ResponsesRequest(input='Hi', store=False,
                                       tools=[{'type': 'function', 'name': 'lookup'}])
        converter = api.StreamConverter(request, [], 'local')
        events = converter.start()
        for event in [ReasoningDelta('Think'), TextDelta('Checking'),
                      ToolCallDelta('call_1', 'lookup', '{}'), DoneEvent('tool_calls')]:
            events += converter.process(EventBatch([event]))
        events += converter.finish()
        data = decoded(events)
        self.assertEqual([e['sequence_number'] for e in data], list(range(len(data))))
        self.assertEqual([e['output_index'] for e in data if e['type'] == 'response.output_item.done'], [0, 1, 2])

    def test_custom_tool_type_and_namespace_stay_stable(self):
        request = api.ResponsesRequest(input='Patch', store=False, tools=[
            {'type': 'namespace', 'name': 'editor', 'tools': [{'type': 'custom', 'name': 'apply_patch'}]}])
        converter = api.StreamConverter(request, [], 'local')
        raw = 'line one\nline two ☃'
        transport = json.dumps({'input': raw})
        events = converter.start()
        for fragment in [transport[:12], transport[12:]]:
            events += converter.process(EventBatch([ToolCallDelta('call_1', 'editor.apply_patch', fragment)]))
        events += converter.process(EventBatch([DoneEvent('tool_calls')]))
        events += converter.finish()
        data = decoded(events)
        items = [e['item'] for e in data if e['type'] in ('response.output_item.added', 'response.output_item.done')]
        self.assertEqual([item['type'] for item in items], ['custom_tool_call'] * 2)
        self.assertEqual(items[0]['id'], items[1]['id'])
        self.assertTrue(all(item['namespace'] == 'editor' and item['name'] == 'apply_patch' for item in items))
        self.assertFalse(any('function_call_arguments' in e['type'] for e in data))
        self.assertEqual(''.join(e['delta'] for e in data if e['type'] == 'response.custom_tool_call_input.delta'), raw)
        self.assertEqual(items[1]['input'], raw)

    def test_structured_output_defaults_and_explicit_limit(self):
        with patch.object(script.shared.args, 'loader', 'ExLlamav3'):
            for explicit, expected in [(None, api.RESPONSES_DEFAULT_STRUCTURED_OUTPUT_TOKENS), (99, 99)]:
                request = api.ResponsesRequest(input='JSON', text=FORMAT, max_output_tokens=explicit, store=False)
                body, _ = api.prepare(request)
                self.assertEqual(body['max_tokens'], expected)
                self.assertIsNotNone(body['_responses_output_grammar'])
            body, _ = api.prepare(api.ResponsesRequest(input='Hello'))
            self.assertEqual(body['max_tokens'], api.RESPONSES_DEFAULT_OUTPUT_TOKENS)

    def test_structured_output_rejects_active_tools_and_validates_answer(self):
        tools = [{'type': 'function', 'name': 'lookup'}]
        with patch.object(script.shared.args, 'loader', 'ExLlamav3'):
            with self.assertRaisesRegex(api.InvalidRequestError, 'active tools'):
                api.prepare(api.ResponsesRequest(input='JSON', text=FORMAT, tools=tools))
            request = api.ResponsesRequest(input='JSON', text=FORMAT, tools=tools, tool_choice='none', store=False)
            api.prepare(request)
            converter = api.StreamConverter(request, [], 'local')
            events = converter.start() + converter.process(EventBatch([TextDelta('{"ok":"wrong"}'), DoneEvent('stop')]))
            from modules.api.responses_format import StructuredOutputError
            with self.assertRaises(StructuredOutputError):
                converter.finish()
            events.append(converter.failed('Invalid output'))
            data = decoded(events)
            self.assertEqual([e['sequence_number'] for e in data], list(range(len(data))))
            self.assertFalse(any(e['type'] == 'response.completed' for e in data))

    def test_owned_generation_lock_is_not_acquired_or_released_twice(self):
        from modules import text_generation
        lock = Mock()
        lock.acquire.side_effect = AssertionError('Nested lock acquisition')
        with patch.object(script.shared, 'generation_lock', lock), \
                patch.object(text_generation.models, 'load_model_if_idle_unloaded'), \
                patch.object(text_generation, '_generate_reply', return_value=iter(['Hello'])):
            self.assertEqual(list(text_generation.generate_reply('Hi', {'_generation_lock_owned': True})), ['Hello'])
        lock.acquire.assert_not_called()
        lock.release.assert_not_called()


class RouteRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_lock_waiter_never_acquires_later(self):
        lock = threading.Lock()
        lock.acquire()
        started = asyncio.Event()
        async def waiter():
            started.set()
            async with script._responses_generation_lock(threading.Event()):
                self.fail('Cancelled waiter entered generation')
        with patch.object(script.shared, 'generation_lock', lock):
            task = asyncio.create_task(waiter())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            lock.release()
            # There is no detached acquisition worker left to steal the lock.
            await asyncio.sleep(0.1)
            self.assertTrue(lock.acquire(blocking=False))
            lock.release()

    async def test_owned_lock_released_on_cancellation_and_error(self):
        lock = threading.Lock()
        entered = asyncio.Event()
        async def owner():
            async with script._responses_generation_lock(threading.Event()) as acquired:
                self.assertTrue(acquired)
                entered.set()
                await asyncio.Event().wait()
        with patch.object(script.shared, 'generation_lock', lock):
            task = asyncio.create_task(owner())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(lock.locked())
            with self.assertRaisesRegex(RuntimeError, 'backend failure'):
                async with script._responses_generation_lock(threading.Event()):
                    raise RuntimeError('backend failure')
            self.assertFalse(lock.locked())

    async def test_schema_tools_rejected_before_generation_over_http(self):
        import httpx
        with patch.object(script.shared.args, 'loader', 'ExLlamav3'), \
                patch.object(script.shared.args, 'api_key', ''), \
                patch.object(script.NativeGeneration, 'stream') as backend:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                for stream in [False, True]:
                    response = await client.post('/v1/responses', json={'input': 'JSON', 'text': FORMAT,
                        'tools': [{'type': 'function', 'name': 'lookup'}], 'stream': stream})
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json()['error']['param'], 'text.format')
            backend.assert_not_called()

    async def test_content_logging_requires_opt_in(self):
        import httpx
        def backend(*args, **kwargs):
            yield EventBatch([ReasoningDelta('PRIVATE_REASONING'), TextDelta('PRIVATE_ANSWER'),
                              ToolCallDelta('call_1', 'lookup', '{"value":"PRIVATE_ARGUMENT"}'), DoneEvent('tool_calls')])
        for enabled in [False, True]:
            with self.subTest(enabled=enabled), patch.object(script, 'RESPONSES_LOG_CONTENT', enabled), \
                    patch.object(script.logger, 'info') as log, \
                    patch.object(script.NativeGeneration, 'stream', side_effect=backend), \
                    patch.object(script.shared.args, 'api_key', ''), \
                    patch('sse_starlette.sse.AppStatus.should_exit_event', None):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                    response = await client.post('/v1/responses', json={'input': 'Hi', 'stream': True, 'store': False,
                        'tools': [{'type': 'function', 'name': 'lookup'}]})
                self.assertIn('response.completed', response.text)
                logged = str(log.call_args_list)
                for value in ['PRIVATE_REASONING', 'PRIVATE_ANSWER', 'PRIVATE_ARGUMENT']:
                    self.assertEqual(value in logged, enabled)
                self.assertIn('stream completed', logged)
