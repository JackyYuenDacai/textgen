"""Exercise cumulative model output through the native Responses pipeline."""
import json
import unittest
from unittest.mock import patch

with patch('sys.argv', ['textgen-tests']):
    from modules.api import generation, responses
from modules.tool_parsing import parse_tool_call


CALL = '<tool_call>{"name":"fetch","arguments":{"url":"one"}}</tool_call>'
SECOND_CALL = '<tool_call>{"name":"fetch","arguments":{"url":"two"}}</tool_call>'


class GenerationToolStreamingTests(unittest.TestCase):
    def run_response(self, chunks, *, stream=True, finish='stop'):
        request = responses.ResponsesRequest(
            input='Fetch the articles', store=False, stream=stream,
            tools=[{'type': 'function', 'name': 'fetch'}],
            max_output_tokens=1000)
        body, history = responses.prepare(request)
        closed = []

        def backend(*args, **kwargs):
            body['_responses_metrics'].update(prompt_tokens=100, completion_tokens=20)
            try:
                for text in chunks:
                    yield {'internal': [['Fetch the articles', text]]}
                body['_responses_metrics']['finish_reason'] = finish
            finally:
                closed.append(True)

        converter = responses.StreamConverter(request, history, 'test')
        events = converter.start()
        with patch.object(generation, 'generate_chat_reply', side_effect=backend):
            for batch in generation.stream(body, stream=stream):
                events.extend(converter.process(batch))
        events.extend(converter.finish())
        self.assertEqual(closed, [True])
        return converter.response, [json.loads(event['data']) for event in events]

    @staticmethod
    def text(response):
        return ''.join(part['text'] for item in response['output'] if item['type'] == 'message'
                       for part in item['content'])

    def test_reasoning_tool_example_does_not_hide_final_answer_or_execute(self):
        thinking = '<think>I considered ' + CALL + ' but no tool is needed.'
        for streaming in (True, False):
            with self.subTest(stream=streaming):
                response, _ = self.run_response([thinking, thinking + '</think>Here is the complete answer.'],
                                                stream=streaming)
                self.assertEqual(self.text(response), 'Here is the complete answer.')
                self.assertFalse(any(item['type'] == 'function_call' for item in response['output']))

    def test_open_reasoning_never_parses_tool_examples(self):
        self.assertEqual(parse_tool_call('<think>Example: ' + CALL, ['fetch']), [])
        self.assertEqual(parse_tool_call('<think>Example: ' + CALL + '</think>', ['fetch']), [])

    def test_parallel_calls_survive_separate_backend_chunks(self):
        for streaming in (True, False):
            with self.subTest(stream=streaming):
                response, events = self.run_response([
                    'Fetching the first batch: ' + CALL,
                    'Fetching the first batch: ' + CALL + '\n' + SECOND_CALL,
                ], stream=streaming)
                calls = [item for item in response['output'] if item['type'] == 'function_call']
                self.assertEqual([json.loads(item['arguments'])['url'] for item in calls], ['one', 'two'])
                self.assertEqual(len({item['call_id'] for item in calls}), 2)
                if streaming:
                    self.assertEqual(self.text(response), 'Fetching the first batch: ')
                self.assertEqual(events[-1]['type'], 'response.completed')

    def test_split_tool_opening_never_leaks_into_text(self):
        answer = 'Fetching: ' + CALL
        response, events = self.run_response([answer[:i] for i in range(1, len(answer) + 1)])
        text = ''.join(event['delta'] for event in events if event['type'] == 'response.output_text.delta')
        self.assertEqual(text, 'Fetching: ')
        self.assertEqual(len([item for item in response['output'] if item['type'] == 'function_call']), 1)

    def test_partial_marker_in_ordinary_text_is_flushed_at_eos(self):
        response, _ = self.run_response(['Value <'])
        self.assertEqual(self.text(response), 'Value <')

    def test_malformed_or_unknown_call_falls_back_to_visible_prose(self):
        for text in ('Fetching: <tool_call>{',
                     'Fetching: <tool_call>{"name":"missing","arguments":{}}</tool_call>',
                     CALL + '\n<tool_call>{'):
            for streaming in (True, False):
                with self.subTest(text=text, stream=streaming):
                    response, _ = self.run_response([text], stream=streaming)
                    self.assertFalse(any(item['type'] == 'function_call' for item in response['output']))

    def test_token_limit_remains_incomplete_and_hides_partial_call(self):
        for streaming in (True, False):
            with self.subTest(stream=streaming):
                response, events = self.run_response(['Fetching: <tool_call>{'], stream=streaming, finish='length')
                self.assertEqual(self.text(response), 'Fetching: ')
                self.assertEqual(events[-1]['type'], 'response.incomplete')
                self.assertEqual(response['incomplete_details'], {'reason': 'max_output_tokens'})

    def test_partial_parallel_batch_at_limit_delivers_no_calls(self):
        response, events = self.run_response([CALL, CALL + '\n<tool_call>{'], finish='length')
        self.assertFalse(any(item['type'] == 'function_call' for item in response['output']))
        self.assertEqual(events[-1]['type'], 'response.incomplete')

    def test_qwen_xml_parallel_calls(self):
        first = '<tool_call><function=fetch><parameter=url>one</parameter></function></tool_call>'
        second = first.replace('>one<', '>two<')
        response, _ = self.run_response(['Fetching: ' + first, 'Fetching: ' + first + second])
        calls = [item for item in response['output'] if item['type'] == 'function_call']
        self.assertEqual([json.loads(item['arguments'])['url'] for item in calls], ['one', 'two'])


class GenerationToolRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_http_parallel_calls_errors_and_token_limit(self):
        import httpx
        from modules.api import script

        for streaming in (True, False):
            for output, reason, expected in (
                (CALL + SECOND_CALL, 'stop', 'tool_calls'),
                ('Fetching: <tool_call>{', 'stop', 'stop'),
                ('Fetching: <tool_call>{', 'length', 'length'),
            ):
                with self.subTest(stream=streaming, expected=expected):
                    def backend(user_input, state, **kwargs):
                        state['_responses_metrics'].update(prompt_tokens=100, completion_tokens=20)
                        # The two valid calls arrive in separate chunks.
                        if expected == 'tool_calls':
                            yield {'internal': [[user_input, CALL]]}
                        yield {'internal': [[user_input, output]]}
                        state['_responses_metrics']['finish_reason'] = reason

                    with patch.object(script.OAIcompletions, 'generate_chat_reply', side_effect=backend), \
                            patch.object(generation, 'generate_chat_reply'), \
                            patch.object(generation, 'parse_tool_call', parse_tool_call), \
                            patch.object(script.shared.args, 'api_key', ''), \
                            patch('sse_starlette.sse.AppStatus.should_exit_event', None):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app),
                                                     base_url='http://127.0.0.1') as client:
                            result = await client.post('/v1/chat/completions', json={
                                'messages': [{'role': 'user', 'content': 'Fetch'}],
                                'tools': [{'type': 'function', 'function': {'name': 'fetch'}}],
                                'stream': streaming, 'max_tokens': 1000})
                    if streaming:
                        self.assertEqual(result.status_code, 200, result.text)
                        lines = [line[6:] for line in result.text.splitlines() if line.startswith('data: ')]
                        self.assertEqual(lines[-1], '[DONE]')
                        chunks = [json.loads(line) for line in lines[:-1]]
                        if expected == 'error':
                            self.assertEqual(chunks[-1]['error']['code'], 'model_output_invalid')
                        else:
                            choice = next(chunk['choices'][0] for chunk in reversed(chunks) if chunk.get('choices'))
                            self.assertEqual(choice['finish_reason'], expected)
                            if expected == 'tool_calls':
                                self.assertEqual(len(choice['delta']['tool_calls']), 2)
                    elif expected == 'error':
                        self.assertEqual(result.status_code, 500)
                        self.assertIn('unrecognized tool call', result.json()['error']['message'])
                    else:
                        self.assertEqual(result.status_code, 200, result.text)
                        self.assertEqual(result.json()['choices'][0]['finish_reason'], expected)

    async def test_http_terminal_events_for_parallel_invalid_and_limited_output(self):
        import httpx
        from modules.api import script

        for output, reason, expected in (
            (CALL + SECOND_CALL, 'stop', 'response.completed'),
            ('Fetching: <tool_call>{', 'stop', 'response.failed'),
            ('Fetching: <tool_call>{', 'length', 'response.incomplete'),
        ):
            with self.subTest(expected=expected):
                def backend(user_input, state, **kwargs):
                    state['_responses_metrics'].update(prompt_tokens=100, completion_tokens=20)
                    yield {'internal': [[user_input, output]]}
                    state['_responses_metrics']['finish_reason'] = reason

                with patch.object(generation, 'generate_chat_reply', side_effect=backend), \
                        patch.object(script.shared.args, 'api_key', ''), \
                        patch('sse_starlette.sse.AppStatus.should_exit_event', None):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app),
                                                 base_url='http://127.0.0.1') as client:
                        result = await client.post('/v1/responses', json={
                            'input': 'Fetch', 'tools': [{'type': 'function', 'name': 'fetch'}],
                            'stream': True, 'store': False})
                self.assertEqual(result.status_code, 200, result.text)
                events = [json.loads(line[6:]) for line in result.text.splitlines() if line.startswith('data: ')]
                self.assertEqual(events[-1]['type'], expected)
                response = events[-1]['response']
                if expected == 'response.completed':
                    self.assertEqual(len([item for item in response['output'] if item['type'] == 'function_call']), 2)
                elif expected == 'response.failed':
                    self.assertEqual(response['error']['code'], 'model_output_invalid')
                    self.assertNotIn('response.completed', [event['type'] for event in events])


if __name__ == '__main__':
    unittest.main()
