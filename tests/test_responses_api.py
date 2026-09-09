import asyncio
import copy
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch('sys.argv', ['textgen-tests']):
    from modules.api import responses as api, script
    from modules.api.errors import InvalidRequestError


def chat_result(content='Hello', calls=None, reason='stop'):
    return {'model': 'loaded-model', 'choices': [{'finish_reason': reason, 'message': {
        'content': content, **({'tool_calls': calls} if calls else {})}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}}


def decoded(events):
    return [json.loads(event['data']) for event in events]


class ResponsesTests(unittest.TestCase):
    def setUp(self):
        self.store = api.ResponseStore()

    def test_text_and_images_conversion_does_not_mutate_input(self):
        raw = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'What is this?'},
                                         {'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='}]}]
        original = copy.deepcopy(raw)
        req = api.ResponsesRequest(input=raw, instructions='Be brief', max_output_tokens=25, store=False)
        body, history = api.prepare(req, self.store)
        self.assertEqual(body['messages'][0], {'role': 'system', 'content': 'Be brief'})
        self.assertEqual(body['messages'][1]['content'][1]['type'], 'image_url')
        self.assertEqual(body['max_tokens'], 25)
        self.assertTrue(body['_responses_no_truncation'])
        self.assertEqual(raw, original)
        self.assertEqual(len(history), 1)

    def test_prior_instructions_not_carried_but_explicit_system_is(self):
        req = api.ResponsesRequest(input=[{'role': 'system', 'content': 'Persistent'}, {'role': 'user', 'content': 'Hi'}], instructions='Old')
        _, history = api.prepare(req, self.store)
        first = api.from_chat(req, chat_result(), history, self.store)
        follow = api.ResponsesRequest(input='Next', instructions='New', previous_response_id=first['id'])
        body, history = api.prepare(follow, self.store)
        self.assertEqual([m['content'] for m in body['messages']], ['New', 'Persistent', 'Hi', 'Hello', 'Next'])
        self.assertNotIn('Old', json.dumps(history))

    def test_tool_roundtrip_ids_and_sender_output(self):
        req = api.ResponsesRequest(input='Weather', tools=[{'type': 'function', 'name': 'weather', 'strict': False}])
        body, history = api.prepare(req, self.store)
        self.assertEqual(body['tools'][0]['function']['name'], 'weather')
        result = api.from_chat(req, chat_result(None, [{'id': 'call_1', 'function': {'name': 'weather', 'arguments': '{"city":"HK"}'}}], 'tool_calls'), history, self.store)
        self.assertEqual(result['output'][0]['call_id'], 'call_1')
        follow = api.ResponsesRequest(previous_response_id=result['id'], input=[{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'Sunny'}])
        body, _ = api.prepare(follow, self.store)
        self.assertEqual(body['messages'][-1], {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Sunny'})
        with self.assertRaises(InvalidRequestError):
            api.prepare(api.ResponsesRequest(previous_response_id=result['id'], input='Ignore'), self.store)

    def test_manual_replay_and_empty_output(self):
        req = api.ResponsesRequest(input='Hi', store=False)
        _, history = api.prepare(req, self.store)
        result = api.from_chat(req, chat_result(''), history, self.store)
        self.assertEqual(result['output'][0]['content'][0]['text'], '')
        body, _ = api.prepare(api.ResponsesRequest(input=[{'role': 'user', 'content': 'Hi'}, *result['output'], {'role': 'user', 'content': 'Next'}]), self.store)
        self.assertEqual(body['messages'][-1]['content'], 'Next')
        with self.assertRaises(InvalidRequestError):
            self.store.get(result['id'])

    def test_rejects_unsupported_and_invalid_inputs(self):
        options = [dict(background=True), dict(tool_choice='required'), dict(parallel_tool_calls=False),
                   dict(text={'format': {'type': 'json_schema'}}), dict(tools=[{'type': 'web_search'}]),
                   dict(tools=[{'type': 'function', 'name': 'x', 'strict': True}]),
                   dict(input=[{'type': 'function_call_output', 'call_id': 'missing', 'output': 'x'}]),
                   dict(input=[{'role': 'user', 'content': [{'type': 'input_file', 'file_id': 'file_1'}]}]),
                   dict(reasoning={'summary': 'invalid'}), dict(input=[{'type': 'reasoning', 'encrypted_content': 'secret'}])]
        for option in options:
            with self.subTest(option=option), self.assertRaises(InvalidRequestError):
                api.prepare(api.ResponsesRequest(**{'input': 'Hi', **option}), self.store)

    def test_storage_expiry_eviction_delete_and_isolation(self):
        now = [0]
        store = api.ResponseStore(max_entries=1, ttl=10, clock=lambda: now[0])
        store.put({'id': 'a'}, [{'content': 'original'}])
        copy_out = store.get('a')
        copy_out['history'][0]['content'] = 'modified'
        self.assertEqual(store.get('a')['history'][0]['content'], 'original')
        store.put({'id': 'b'}, [])
        with self.assertRaises(InvalidRequestError): store.get('a')
        now[0] = 11
        with self.assertRaises(InvalidRequestError): store.get('b')
        self.assertEqual(store.bytes, 0)
        store.put({'id': 'c'}, [])
        self.assertTrue(store.delete('c')['deleted'])
        with self.assertRaises(InvalidRequestError): store.get('c')

    def test_codex_metadata_reasoning_hints_and_namespace_roundtrip(self):
        request = api.ResponsesRequest(input='Hi', client_metadata={'session_id': 'test'},
            include=['reasoning.encrypted_content'], reasoning={'summary': 'auto', 'effort': 'high'},
            text={'verbosity': 'low'}, tools=[{'type': 'namespace', 'name': 'files', 'description': 'Files',
                'tools': [{'type': 'function', 'name': 'read', 'strict': False}]}])
        body, history = api.prepare(request, self.store)
        self.assertEqual(body['tools'][0]['function']['name'], 'files.read')
        self.assertNotIn('client_metadata', body)
        self.assertEqual(body['reasoning_effort'], 'high')
        result = api.from_chat(request, chat_result(None, [{'id': 'call_n', 'function': {
            'name': 'files.read', 'arguments': '{}'}}], 'tool_calls'), history, self.store)
        self.assertEqual(result['output'][0]['namespace'], 'files')
        self.assertEqual(result['output'][0]['name'], 'read')
        replay = api.input_messages(result['output'])
        self.assertEqual(replay[0]['tool_calls'][0]['function']['name'], 'files.read')
        self.assertIsNone(result['reasoning']['summary'])
        self.assertNotIn('encrypted_content', json.dumps(result))

    def test_history_preserves_reasoning_calls_and_empty_tool_result_in_prompt(self):
        from modules.chat import generate_chat_prompt
        items = [{'role': 'user', 'content': 'Question'},
                 {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'My reasoning'}]},
                 {'role': 'assistant', 'content': 'Checking'},
                 {'type': 'function_call', 'call_id': 'call_a', 'name': 'check', 'arguments': '{}'},
                 {'type': 'function_call_output', 'call_id': 'call_a', 'output': ''}]
        body, _ = api.prepare(api.ResponsesRequest(input=items), self.store)
        user, system, history = script.OAIcompletions.convert_history(body['messages'], True)
        template = '{% for m in messages %}{{ m | tojson }}\n{% endfor %}'
        state = dict(script.shared.settings, mode='instruct', history=history,
                     custom_system_message=system, instruction_template_str=template,
                     chat_template_str=template)
        with patch.object(script.shared, 'tokenizer', None):
            prompt = generate_chat_prompt(user, state)
        rendered = [json.loads(line) for line in prompt.splitlines() if line.strip()]
        assistant = next(m for m in rendered if m['role'] == 'assistant')
        self.assertEqual(assistant['reasoning_content'], 'My reasoning')
        self.assertEqual(assistant['content'], 'Checking')
        self.assertEqual(assistant['tool_calls'][0]['id'], 'call_a')
        self.assertEqual(rendered[-1], {'role': 'tool', 'tool_call_id': 'call_a', 'content': ''})

    def test_prompt_overflow_rejected_before_clipping(self):
        from modules.chat import generate_chat_prompt
        template = '{% for m in messages %}{{ m.content }}{% endfor %}'
        state = dict(script.shared.settings, mode='instruct', history={'internal': [], 'visible': []},
                     instruction_template_str=template, chat_template_str=template,
                     _responses_no_truncation=True)
        with patch.object(script.shared, 'tokenizer', object()), \
                patch('modules.chat.get_encoded_length', return_value=100), \
                patch('modules.chat.get_max_prompt_length', return_value=50):
            with self.assertRaises(InvalidRequestError):
                generate_chat_prompt('Long prompt', state)

    def test_custom_backend_errors_propagate_only_for_responses(self):
        from modules.text_generation import generate_reply_custom
        model = SimpleNamespace(generate=Mock(side_effect=RuntimeError('backend failed')),
                                last_prompt_token_count=3)
        with patch.object(script.shared, 'model', model), patch('modules.text_generation.set_manual_seed', return_value=1), \
                patch('modules.text_generation.logger'):
            with self.assertRaisesRegex(RuntimeError, 'backend failed'):
                list(generate_reply_custom('q', 'q', dict(seed=1, stream=False, _responses_raise_errors=True), is_chat=True))
            self.assertEqual(list(generate_reply_custom('q', 'q', dict(seed=1, stream=False), is_chat=True)), [])

    def test_stream_text_lifecycle_sequence_and_usage(self):
        req = api.ResponsesRequest(input='Hi', stream=True)
        _, history = api.prepare(req, self.store)
        stream = api.StreamConverter(req, history, 'model', self.store)
        events = stream.start()
        for piece in ['He', 'llo']:
            events += stream.process({'choices': [{'delta': {'content': piece}}]})
        events += stream.process({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        events += stream.process({'choices': [], 'usage': chat_result()['usage']})
        events += stream.finish()
        data = decoded(events)
        self.assertEqual([e['sequence_number'] for e in data], list(range(len(data))))
        self.assertEqual(data[0]['response']['output'], [])
        self.assertEqual(''.join(e['delta'] for e in data if e['type'] == 'response.output_text.delta'), 'Hello')
        result = data[-1]['response']
        self.assertEqual(data[-1]['type'], 'response.completed')
        self.assertEqual(result['usage']['input_tokens'], 10)
        self.assertEqual(self.store.get(result['id'])['response'], result)

    def test_stream_tools_buffer_raw_markup_and_assemble_arguments(self):
        req = api.ResponsesRequest(input='Hi', tools=[{'type': 'function', 'name': 'f', 'strict': False}], store=False)
        stream = api.StreamConverter(req, [], 'model', self.store)
        events = stream.start()
        events += stream.process({'choices': [{'delta': {'content': '<tool_call>'}}]})
        events += stream.process({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'f', 'arguments': '{'}}]}}]})
        events += stream.process({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '}'}}]}, 'finish_reason': 'tool_calls'}]})
        events += stream.finish()
        data = decoded(events)
        self.assertNotIn('response.output_text.delta', [e['type'] for e in data])
        self.assertEqual(data[-1]['response']['output'][0]['arguments'], '{}')
        self.assertEqual(data[-1]['response']['output'][0]['call_id'], 'call_1')

    def test_reasoning_and_length_are_not_fake_completion(self):
        req = api.ResponsesRequest(input='Hi', store=False)
        stream = api.StreamConverter(req, [], 'model', self.store)
        events = stream.start() + stream.process({'choices': [{'delta': {'reasoning_content': 'Thinking'}, 'finish_reason': 'length'}]})
        events += stream.finish()
        result = decoded(events)[-1]
        self.assertEqual(result['type'], 'response.incomplete')
        self.assertEqual(result['response']['output'][0]['summary'], [])
        self.assertEqual(result['response']['incomplete_details']['reason'], 'max_output_tokens')
        self.assertIsNone(result['response']['usage'])


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_stream_failure_never_completed_or_stored(self):
        import httpx
        def backend(*args, **kwargs):
            yield {'choices': [{'delta': {'content': 'Partial'}}]}
            raise InvalidRequestError('Context overflow', 'input')
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
                patch.object(script.shared.args, 'api_key', ''), patch.object(api.STORE, 'put') as put, \
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                result = await client.post('/v1/responses', json={'input': 'Hi', 'stream': True})
            self.assertEqual(result.status_code, 200, result.text)
            events = [json.loads(line[6:]) for line in result.text.splitlines() if line.startswith('data: ')]
            self.assertEqual(events[-1]['type'], 'response.failed')
            self.assertEqual(events[-1]['response']['error']['code'], 'invalid_prompt')
            self.assertNotIn('response.completed', [e['type'] for e in events])
            put.assert_not_called()

    async def test_stream_disconnect_cancels_prefill_without_persisting(self):
        entered, finished = threading.Event(), threading.Event()
        def backend(*args, stop_event, **kwargs):
            entered.set()
            try:
                if not stop_event.wait(3):
                    yield {'choices': []}
            finally:
                finished.set()
        request = SimpleNamespace()
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
                patch('sse_starlette.sse.AppStatus.should_exit_event', None), patch.object(api.STORE, 'put') as put:
            result = await script.openai_responses(request, api.ResponsesRequest(input='Hi', stream=True))
            async def receive():
                while not entered.is_set(): await asyncio.sleep(.001)
                return {'type': 'http.disconnect'}
            async def send(message): pass
            try:
                await asyncio.wait_for(result({'type': 'http'}, receive, send), 2)
                self.assertTrue(finished.is_set())
                self.assertTrue(result.stop_event.is_set())
                put.assert_not_called()
            finally:
                result.stop_event.set()
                await result.body_iterator.aclose()

    async def test_http_shapes_auth_validation_and_storage(self):
        import httpx
        transport = httpx.ASGITransport(app=script.app)
        with patch.object(script.OAIcompletions, 'chat_completions', return_value=chat_result()), \
                patch.object(script.shared.args, 'api_key', 'test-secret'):
            async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
                response = await client.post('/v1/responses', json={'input': 'Hi'})
                self.assertEqual(response.status_code, 401)
                headers = {'Authorization': 'Bearer test-secret'}
                response = await client.post('/v1/responses', headers=headers, json={'input': 'Hi', 'unknown_field': True})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()['error']['param'], 'unknown_field')
                self.assertIn('unknown_field', response.json()['error']['message'])
                response = await client.post('/v1/responses', headers=headers, json={'input': 'Hi'})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                retrieved = await client.get('/v1/responses/' + body['id'], headers=headers)
                self.assertEqual(retrieved.json(), body)
                removed = await client.delete('/v1/responses/' + body['id'], headers=headers)
                self.assertTrue(removed.json()['deleted'])
                self.assertEqual((await client.get('/v1/responses/' + body['id'], headers=headers)).status_code, 404)


if __name__ == '__main__':
    unittest.main()
