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
        options = [dict(background=True), dict(tool_choice='required'),
                   dict(text={'format': {'type': 'json_schema'}}), dict(tools=[{'type': 'web_search'}]),
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

    def test_response_lookup_does_not_copy_history_and_snapshot_is_isolated(self):
        self.store.put({'id': 'r', 'output': []}, [{'content': 'large history'}])
        original_copy = copy.deepcopy
        observed = []
        def copying(value, *args, **kwargs):
            observed.append(value)
            return original_copy(value, *args, **kwargs)
        with patch.object(api.copy, 'deepcopy', side_effect=copying):
            result = self.store.get('r', field='response')
        self.assertEqual(observed, [{'id': 'r', 'output': []}])
        result['output'].append('changed')
        self.assertEqual(self.store.get('r', field='response')['output'], [])
        self.store.put({'id': 'r', 'output': ['new']}, [])
        size = len(json.dumps(self.store.get('r'), ensure_ascii=False, separators=(',', ':')).encode())
        self.assertEqual(self.store.bytes, size)

    def test_tool_turn_cannot_skip_unresolved_results(self):
        with self.assertRaises(InvalidRequestError):
            api.prepare(api.ResponsesRequest(input=[
                {'type': 'function_call', 'call_id': 'c', 'name': 'f', 'arguments': '{}'},
                {'role': 'user', 'content': 'New turn before result'},
                {'type': 'function_call_output', 'call_id': 'c', 'output': 'done'}]), self.store)

    def test_input_items_paginate_repeated_messages_in_both_orders(self):
        raw = [{'role': 'user', 'content': 'Again'} for _ in range(23)]
        request = api.ResponsesRequest(input=raw, instructions='Separate instructions')
        _, history = api.prepare(request, self.store)
        result = api.from_chat(request, chat_result('Not an input'), history, self.store)
        all_items = self.store.input_items(result['id'], order='asc', limit=100)['data']
        ids = [item['id'] for item in all_items]
        self.assertEqual(len(set(ids)), 23)
        self.assertEqual(all_items[0]['content'], [{'type': 'input_text', 'text': 'Again'}])
        self.assertNotIn('Not an input', json.dumps(all_items))
        self.assertNotIn('Separate instructions', json.dumps(all_items))
        self.assertEqual(raw, [{'role': 'user', 'content': 'Again'} for _ in range(23)])
        default_page = self.store.input_items(result['id'])
        self.assertEqual([item['id'] for item in default_page['data']], ids[::-1][:20])
        self.assertTrue(default_page['has_more'])
        for order, expected in [('asc', ids), ('desc', ids[::-1])]:
            collected, cursor = [], None
            while True:
                page = self.store.input_items(result['id'], after=cursor, order=order, limit=5)
                self.assertEqual(page['first_id'], page['data'][0]['id'])
                self.assertEqual(page['last_id'], page['data'][-1]['id'])
                collected.extend(item['id'] for item in page['data'])
                cursor = page['last_id']
                if not page['has_more']:
                    break
            self.assertEqual(collected, expected)
            self.assertEqual(self.store.input_items(result['id'], after=cursor, order=order), {
                'object': 'list', 'data': [], 'first_id': None, 'last_id': None, 'has_more': False})
        for options in ({'after': ''}, {'after': result['output'][0]['id']}, {'limit': 0},
                        {'limit': 101}, {'order': 'sideways'}, {'include': ['file_search_call.results']}):
            with self.subTest(options=options), self.assertRaises(InvalidRequestError):
                self.store.input_items(result['id'], **options)

    def test_input_items_preserve_custom_tools_reasoning_images_and_replay(self):
        raw = [
            {'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='}]},
            {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'Inspect it'}]},
            {'role': 'assistant', 'phase': 'commentary', 'content': 'Checking'},
            {'type': 'custom_tool_call', 'call_id': 'patch', 'namespace': 'editor', 'name': 'patch', 'input': 'raw code'},
            {'type': 'custom_tool_call_output', 'call_id': 'patch', 'output': ''},
            {'type': 'function_call', 'call_id': 'read', 'name': 'read', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'read', 'output': [
                {'type': 'input_text', 'text': 'Result'}, {'type': 'input_image', 'image_url': 'data:image/png;base64,BB=='}]},
        ]
        original = copy.deepcopy(raw)
        request = api.ResponsesRequest(input=raw)
        body, history = api.prepare(request, self.store)
        result = api.from_chat(request, chat_result(), history, self.store)
        page = self.store.input_items(result['id'], order='asc', include=['message.input_image.image_url'])
        items = page['data']
        self.assertEqual([item['type'] for item in items], [
            'message', 'reasoning', 'message', 'custom_tool_call', 'custom_tool_call_output',
            'function_call', 'function_call_output'])
        self.assertEqual(items[1]['content'], raw[1]['content'])
        self.assertEqual(items[2]['phase'], 'commentary')
        self.assertEqual(items[3]['namespace'], 'editor')
        self.assertEqual(items[3]['input'], 'raw code')
        self.assertEqual(items[4]['output'], '')
        self.assertEqual(items[6]['output'], raw[6]['output'])
        self.assertEqual(len({item['id'] for item in items}), len(raw))
        replay, _ = api.prepare(api.ResponsesRequest(input=items), self.store)
        self.assertEqual(replay['messages'], body['messages'])
        self.assertEqual(raw, original)

    def test_input_items_chain_survives_parent_delete_and_preserves_ids(self):
        req = api.ResponsesRequest(input='First', instructions='Old')
        _, history = api.prepare(req, self.store)
        first = api.from_chat(req, chat_result('First answer'), history, self.store)
        first_inputs = self.store.input_items(first['id'], order='asc')['data']
        follow = api.ResponsesRequest(input='Next', previous_response_id=first['id'], instructions='New')
        _, history = api.prepare(follow, self.store)
        self.store.delete(first['id'])
        second = api.from_chat(follow, chat_result('Second answer'), history, self.store)
        items = self.store.input_items(second['id'], order='asc')['data']
        self.assertEqual(items[:2], first_inputs + first['output'])
        self.assertEqual(items[2]['content'][0]['text'], 'Next')
        self.assertEqual(len(items), 3)
        with self.assertRaises(InvalidRequestError):
            self.store.input_items(second['id'], after=second['output'][0]['id'])
        # Stored output item IDs remain stable even after their parent is gone.
        self.assertEqual(self.store.input_items(second['id'], after=first['output'][0]['id'], order='asc')['data'], items[2:])

    def test_input_items_copy_only_page_and_obey_storage_lifetime(self):
        now = [0]
        store = api.ResponseStore(max_entries=1, ttl=10, clock=lambda: now[0])
        request = api.ResponsesRequest(input=[{'role': 'user', 'content': str(i)} for i in range(30)])
        _, history = api.prepare(request, store)
        result = api.from_chat(request, chat_result(), history, store)
        with patch.object(api.copy, 'deepcopy', wraps=copy.deepcopy) as copying:
            page = store.input_items(result['id'], limit=1)
        self.assertEqual(copying.call_count, 1)
        self.assertEqual(len(copying.call_args.args[0]), 1)
        page['data'][0]['content'][0]['text'] = 'Mutated'
        self.assertEqual(store.input_items(result['id'], limit=1)['data'][0]['content'][0]['text'], '29')
        size = len(json.dumps(store.get(result['id']), ensure_ascii=False, separators=(',', ':')).encode())
        self.assertEqual(store.bytes, size)
        now[0] = 11
        with self.assertRaises(InvalidRequestError):
            store.input_items(result['id'])
        self.assertEqual(store.bytes, 0)
        result = api.from_chat(request, chat_result(), history, store)
        newer = api.from_chat(request, chat_result(), history, store)
        with self.assertRaises(InvalidRequestError):
            store.input_items(result['id'])
        store.delete(newer['id'])
        with self.assertRaises(InvalidRequestError):
            store.input_items(newer['id'])
        request.store = False
        transient = api.from_chat(request, chat_result(), history, store)
        with self.assertRaises(InvalidRequestError):
            store.input_items(transient['id'])

    def test_input_item_ids_validate_and_page_can_be_replayed(self):
        for raw in ([{'role': 'user', 'content': 'Hi', 'id': 42}],
                    [{'role': 'user', 'content': 'Hi', 'id': 'same'}] * 2):
            with self.assertRaises(InvalidRequestError):
                api.prepare(api.ResponsesRequest(input=raw), self.store)
        request = api.ResponsesRequest(input=[{'role': 'assistant', 'content': 'Hello', 'id': 'msg_original'}])
        body, history = api.prepare(request, self.store)
        result = api.from_chat(request, chat_result(), history, self.store)
        items = self.store.input_items(result['id'])['data']
        self.assertEqual(items[0]['id'], 'msg_original')
        self.assertEqual(items[0]['content'][0]['type'], 'output_text')
        replay, _ = api.prepare(api.ResponsesRequest(input=items), self.store)
        self.assertEqual(body['messages'], replay['messages'])

    def test_prompt_cache_options_do_not_change_model_input(self):
        a, _ = api.prepare(api.ResponsesRequest(input='Stable prompt', client_metadata={'turn': 'a'},
                           prompt_cache_key='one', prompt_cache_retention='in-memory'), self.store)
        b, _ = api.prepare(api.ResponsesRequest(input='Stable prompt', client_metadata={'turn': 'b'},
                           prompt_cache_key='two'), self.store)
        a.pop('_responses_metrics'); b.pop('_responses_metrics')
        self.assertEqual(a, b)
        with self.assertRaises(InvalidRequestError):
            api.prepare(api.ResponsesRequest(input='Hi', prompt_cache_retention='24h'), self.store)

    def test_custom_tool_raw_input_stream_and_replay(self):
        raw = '*** Begin Patch\n*** End Patch\n'
        request = api.ResponsesRequest(input='Patch', tools=[{'type': 'namespace', 'name': 'editor', 'tools': [
            {'type': 'custom', 'name': 'apply_patch', 'format': {'type': 'grammar', 'syntax': 'lark',
             'definition': 'start: "*** Begin Patch" NEWLINE "*** End Patch" NEWLINE\n%import common.NEWLINE'}}]}])
        body, history = api.prepare(request, self.store)
        self.assertEqual(body['tools'][0]['function']['parameters']['required'], ['input'])
        stream = api.StreamConverter(request, history, 'local', self.store)
        events = stream.start() + stream.process({'choices': [{'delta': {'tool_calls': [
            {'index': 0, 'id': 'patch1', 'function': {'name': 'editor.apply_patch', 'arguments': json.dumps({'input': raw})}}]},
            'finish_reason': 'tool_calls'}]}) + stream.finish()
        data = decoded(events)
        self.assertIn('response.custom_tool_call_input.delta', [e['type'] for e in data])
        item = data[-1]['response']['output'][0]
        self.assertEqual(item['type'], 'custom_tool_call')
        self.assertEqual(item['input'], raw)
        self.assertEqual(item['namespace'], 'editor')
        follow, _ = api.prepare(api.ResponsesRequest(previous_response_id=data[-1]['response']['id'], input=[
            {'type': 'custom_tool_call_output', 'call_id': 'patch1', 'output': 'Success'}]), self.store)
        call = follow['messages'][-2]['tool_calls'][0]
        self.assertEqual(json.loads(call['function']['arguments'])['input'], raw)
        self.assertEqual(follow['messages'][-1]['content'], 'Success')
        with self.assertRaises(ValueError):
            api.output_call(request, 'bad', 'editor.apply_patch', '{"input":"invalid patch"}')

    def test_strict_tool_outputs_are_checked_before_exposing_calls(self):
        request = api.ResponsesRequest(input='Read', store=False, tools=[{'type': 'function', 'name': 'read', 'strict': True,
            'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id'], 'additionalProperties': False}}])
        api.prepare(request, self.store)
        self.assertEqual(api.output_call(request, 'ok', 'read', '{"id":1}')['arguments'], '{"id":1}')
        with self.assertRaises(ValueError):
            api.output_call(request, 'bad', 'read', '{"id":"text"}')
        with self.assertRaises(InvalidRequestError):
            api.prepare(api.ResponsesRequest(input='Read', tools=[{'type': 'function', 'name': 'read', 'strict': True,
                'parameters': {'type': 'object', '$ref': 'https://example.com/schema'}}]), self.store)

    def test_assistant_phase_survives_manual_and_stored_history(self):
        items = [{'role': 'assistant', 'phase': 'commentary', 'content': 'Checking'},
                 {'role': 'assistant', 'phase': 'final_answer', 'content': 'Done'}]
        messages = api.input_messages(items)
        self.assertEqual(len(messages), 2)
        _, _, history = script.OAIcompletions.convert_history(messages, True)
        self.assertEqual([entry[3]['response_assistant']['phase'] for entry in history['internal']], ['commentary', 'final_answer'])

    def test_custom_regex_and_import_restrictions(self):
        req = api.ResponsesRequest(input='Run', tools=[{'type': 'custom', 'name': 'code',
            'format': {'type': 'grammar', 'syntax': 'regex', 'definition': '[0-9]+'}}])
        api.prepare(req, self.store)
        self.assertEqual(api.output_call(req, 'x', 'code', '{"input":"123"}')['input'], '123')
        with self.assertRaises(ValueError):
            api.output_call(req, 'x', 'code', '{"input":"abc"}')
        with self.assertRaises(InvalidRequestError):
            api.prepare(api.ResponsesRequest(input='Run', tools=[{'type': 'custom', 'name': 'code',
                'format': {'type': 'grammar', 'syntax': 'lark', 'definition': '%import .private.VALUE\nstart: VALUE'}}]), self.store)

    def test_tool_backend_closed_before_terminal_chunk(self):
        body, _ = api.prepare(api.ResponsesRequest(input='Read', tools=[{'type': 'function', 'name': 'f'}], stream=True), self.store)
        closed = threading.Event()
        def backend(*args, **kwargs):
            body['_responses_metrics'].update(prompt_tokens=12, completion_tokens=4)
            try:
                yield {'internal': [['Read', 'tool markup']]}
                raise AssertionError('Should stop on the recognized call')
            finally:
                closed.set()
        with patch.object(script.OAIcompletions, 'generate_chat_reply', side_effect=backend), \
                patch.object(script.OAIcompletions, 'parse_tool_call', return_value=[
                    {'type': 'function', 'function': {'name': 'f', 'arguments': {}}}]):
            generator = script.OAIcompletions.stream_chat_completions(body)
            try:
                next(generator)  # Initial role chunk.
                final = next(generator)
                self.assertEqual(final['choices'][0]['finish_reason'], 'tool_calls')
                self.assertTrue(closed.is_set())
            finally:
                generator.close()

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
        self.assertEqual(self.store.input_items(result['id'])['data'][0]['content'][0]['text'], 'Hi')

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

    def test_tool_batch_validation_never_exposes_partial_executable_output(self):
        tool = {'type': 'function', 'name': 'f', 'strict': True,
                'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}
        request = api.ResponsesRequest(input='Hi', tools=[tool], store=False, parallel_tool_calls=False)
        body, _ = api.prepare(request)
        self.assertIn('at most one tool', body['messages'][0]['content'])
        calls = [{'id': 'a', 'name': 'f', 'arguments': '{}'}, {'id': 'b', 'name': 'f', 'arguments': '{}'}]
        with self.assertRaisesRegex(api.ToolOutputError, 'multiple calls'):
            api.output_calls(request, calls)
        request.parallel_tool_calls = True
        for bad in ({'name': 'unknown'}, {'arguments': '{"extra":1}'}, {'id': 'a'}):
            stream = api.StreamConverter(request, [], 'model')
            stream.calls = {0: calls[0], 1: {**calls[1], **bad}}
            stream.finish_reason = 'tool_calls'
            with self.assertRaises(api.ToolOutputError):
                stream.finish()
            self.assertEqual(stream.response['output'], [])


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


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_cancellation_and_exception_release_without_workers(self):
        queue = api.GenerationQueue()
        release = asyncio.Event()
        order = []
        stops = [threading.Event() for _ in range(48)]
        async def run(index):
            async with queue.slot(stops[index]) as admitted:
                if not admitted: return
                order.append(index)
                if index == 0: await release.wait()
                if index == 47: raise RuntimeError('expected')
        tasks = [asyncio.create_task(run(i)) for i in range(48)]
        try:
            await asyncio.sleep(.02)
            self.assertEqual(len(queue.waiters), 48)
            self.assertEqual(order, [0])
            stops[1].set()
            tasks[2].cancel()
            await asyncio.sleep(.15)
            self.assertEqual(len(queue.waiters), 46)
            self.assertEqual(order, [0])
        finally:
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        self.assertEqual(order, [0] + list(range(3,48)))
        self.assertIsInstance(results[2], asyncio.CancelledError)
        self.assertIsInstance(results[47], RuntimeError)
        self.assertEqual(len(queue.waiters), 0)
        async with queue.slot(threading.Event()) as admitted:
            self.assertTrue(admitted)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_nonstream_keeps_slot_until_worker_exits(self):
        entered, stopping, release = threading.Event(), threading.Event(), threading.Event()
        second_entered = threading.Event()
        calls = []
        def backend(body, *, stop_event):
            calls.append(body['messages'][-1]['content'])
            if len(calls) == 1:
                entered.set()
                if not stop_event.wait(3): raise TimeoutError('cancellation not signalled')
                stopping.set()
                if not release.wait(3): raise TimeoutError('cleanup not released')
            else:
                second_entered.set()
            return chat_result()
        async def receive():
            await asyncio.Event().wait()
        request = SimpleNamespace(receive=receive)
        first = second = None
        with patch.object(api, 'GENERATION_QUEUE', api.GenerationQueue()), \
                patch.object(script.OAIcompletions, 'chat_completions', side_effect=backend):
            try:
                first = asyncio.create_task(script.openai_responses(request, api.ResponsesRequest(input='first', store=False)))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                first.cancel()
                self.assertTrue(await asyncio.to_thread(stopping.wait, 2))
                second = asyncio.create_task(script.openai_responses(request, api.ResponsesRequest(input='second', store=False)))
                await asyncio.sleep(.05)
                self.assertFalse(second_entered.is_set())
                release.set()
                outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 3)
                self.assertIsInstance(outcomes[0], asyncio.CancelledError)
                self.assertEqual(outcomes[1].status_code, 200)
                self.assertEqual(calls, ['first', 'second'])
            finally:
                release.set()
                for task in (first, second):
                    if task and not task.done(): task.cancel()

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
