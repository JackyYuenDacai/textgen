"""Optional SDK integration tests; no live server, GPU, or external requests.

Run with openai installed in a test environment. Production does not require it.
"""
import json
import unittest
from unittest.mock import patch

try:
    import openai
except ImportError:
    openai = None
else:
    # SDK 3 uses httpx2; SDK 1/2 use httpx.
    from openai import _base_client
    httpx = getattr(_base_client, 'httpx2', None) or _base_client.httpx

from test_responses_api import api, script, chat_result


@unittest.skipIf(openai is None, 'Optional openai SDK is not installed')
class SDKTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app))
        self.client = openai.AsyncOpenAI(api_key='test', base_url='http://127.0.0.1/v1', http_client=self.http)
        self.auth = patch.object(script.shared.args, 'api_key', 'test')
        self.auth.start()
        self.addCleanup(self.auth.stop)

    async def asyncTearDown(self):
        await self.client.close()

    async def test_sdk_create_tool_result_retrieve_delete(self):
        calls = [{'id': 'call_1', 'function': {'name': 'weather', 'arguments': '{"city":"HK"}'}}]
        with patch.object(script.OAIcompletions, 'chat_completions', return_value=chat_result(None, calls, 'tool_calls')):
            first = await self.client.responses.create(model='local', input='Weather',
                tools=[{'type': 'function', 'name': 'weather', 'strict': False}])
        self.assertEqual(first.output[0].call_id, 'call_1')
        with patch.object(script.OAIcompletions, 'chat_completions', return_value=chat_result('Sunny')) as backend:
            result = await self.client.responses.create(model='local', previous_response_id=first.id,
                input=[{'type': 'function_call_output', 'call_id': first.output[0].call_id, 'output': 'Sunny'}])
        self.assertEqual(result.output_text, 'Sunny')
        self.assertEqual(backend.call_args.args[0]['messages'][-1]['role'], 'tool')
        self.assertEqual((await self.client.responses.retrieve(result.id)).output_text, 'Sunny')
        await self.client.responses.delete(result.id)
        with self.assertRaises(openai.NotFoundError):
            await self.client.responses.retrieve(result.id)
        await self.client.responses.delete(first.id)

    async def test_sdk_stream_accumulates_reasoning_and_text(self):
        def backend(*args, **kwargs):
            yield {'choices': [{'delta': {'reasoning_content': 'Think'}}]}
            yield {'choices': [{'delta': {'content': 'Hel'}}]}
            yield {'choices': [{'delta': {'content': 'lo'}, 'finish_reason': 'stop'}]}
            yield {'choices': [], 'usage': chat_result()['usage']}
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with self.client.responses.stream(model='local', input='Hi', store=False) as stream:
                events = [event async for event in stream]
                result = await stream.get_final_response()
        self.assertEqual(result.output_text, 'Hello')
        self.assertEqual(result.output[0].content[0].text, 'Think')
        self.assertEqual(result.usage.total_tokens, 13)
        self.assertEqual(result.status, 'completed')
        self.assertIn('response.output_text.delta', [event.type for event in events])

    async def test_sdk_custom_tool_stream_and_stateless_replay(self):
        raw = '*** Begin Patch\n*** Add File: result.txt\n+hello\n*** End Patch'
        tool = {'type': 'custom', 'name': 'apply_patch', 'format': {'type': 'text'}}
        def backend(*args, **kwargs):
            yield {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'patch_1',
                'function': {'name': 'apply_patch', 'arguments': json.dumps({'input': raw})}}]},
                'finish_reason': 'tool_calls'}]}
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with self.client.responses.stream(model='local', input='Create result.txt',
                    tools=[tool], parallel_tool_calls=False, store=False) as stream:
                events = [event async for event in stream]
                result = await stream.get_final_response()
        self.assertEqual(result.output[0].type, 'custom_tool_call')
        self.assertEqual(result.output[0].input, raw)
        self.assertIn('response.custom_tool_call_input.done', [e.type for e in events])
        with patch.object(script.OAIcompletions, 'chat_completions', return_value=chat_result('Done')) as backend:
            follow = await self.client.responses.create(model='local', store=False, tools=[tool], input=[
                {'role': 'user', 'content': 'Create result.txt'}, result.output[0].model_dump(exclude_none=True),
                {'type': 'custom_tool_call_output', 'call_id': 'patch_1', 'output': 'Success'}])
        self.assertEqual(follow.output_text, 'Done')
        messages = backend.call_args.args[0]['messages']
        self.assertEqual(json.loads(messages[-2]['tool_calls'][0]['function']['arguments'])['input'], raw)

    async def test_invalid_generated_tool_is_a_useful_failure_not_executable(self):
        tool = {'type': 'function', 'name': 'read', 'strict': True, 'parameters': {
            'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']}}
        calls = [{'id': 'bad', 'function': {'name': 'read', 'arguments': '{"id":"wrong"}'}}]
        with patch.object(script.OAIcompletions, 'chat_completions', return_value=chat_result(None, calls, 'tool_calls')):
            # Disable SDK retries for this deterministic invalid model result.
            with self.assertRaisesRegex(openai.InternalServerError, 'strict schema validation'):
                await self.client.with_options(max_retries=0).responses.create(model='local', input='Read', tools=[tool])
        def backend(*args, **kwargs):
            yield {'choices': [{'delta': {'tool_calls': [{'index': 0, **calls[0]}]}, 'finish_reason': 'tool_calls'}]}
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            stream = await self.client.responses.create(model='local', input='Read', tools=[tool], stream=True)
            events = [event async for event in stream]
        self.assertEqual(events[-1].type, 'response.failed')
        self.assertEqual(events[-1].response.error.code, 'model_output_invalid')
        self.assertEqual(events[-1].response.output, [])
        self.assertNotIn('response.output_item.done', [event.type for event in events])


    async def test_sdk_stream_tools_and_incomplete_response(self):
        for finish_reason in ('tool_calls', 'length'):
            def backend(*args, **kwargs):
                if finish_reason == 'tool_calls':
                    for index in range(2):
                        yield {'choices': [{'delta': {'tool_calls': [{'index': index, 'id': f'call_{index}',
                            'function': {'name': 'lookup', 'arguments': json.dumps({'index': index})}}]}}]}
                else:
                    yield {'choices': [{'delta': {'content': 'Cut short'}}]}
                yield {'choices': [{'delta': {}, 'finish_reason': finish_reason}]}
            with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
                    patch('sse_starlette.sse.AppStatus.should_exit_event', None):
                async with self.client.responses.stream(model='local', input='Hi', store=False,
                        tools=[{'type': 'function', 'name': 'lookup', 'strict': False}]) as stream:
                    events = [event async for event in stream]
                    if finish_reason == 'length':
                        # SDK 3.10's helper only returns response.completed. Consume
                        # the protocol's incomplete terminal event directly.
                        result = next(e.response for e in events if e.type == 'response.incomplete')
                    else:
                        result = await stream.get_final_response()
            if finish_reason == 'tool_calls':
                self.assertEqual([item.call_id for item in result.output], ['call_0', 'call_1'])
                self.assertEqual(result.output[1].arguments, '{"index": 1}')
            else:
                self.assertEqual(result.status, 'incomplete')
                self.assertEqual(result.incomplete_details.reason, 'max_output_tokens')


if __name__ == '__main__':
    unittest.main()
