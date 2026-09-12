path = r'F:\GitHub\textgen\tests\test_responses_api.py'
src = open(path, encoding='utf-8').read()

old = "import asyncio\nimport copy\nimport json\nimport threading\nimport unittest\n"
new = "import asyncio\nimport copy\nimport json\nimport threading\nimport time\nimport unittest\n"
assert src.count(old) == 1
src = src.replace(old, new)

old = """            self.assertEqual(events[-1]['type'], 'response.failed')
            self.assertEqual(events[-1]['response']['error']['code'], 'invalid_prompt')
            self.assertNotIn('response.completed', [e['type'] for e in events])
            put.assert_not_called()
"""
new = """            self.assertEqual(events[-1]['type'], 'response.failed')
            self.assertEqual(events[-1]['response']['error']['code'], 'invalid_prompt')
            self.assertNotIn('response.completed', [e['type'] for e in events])
            put.assert_not_called()

    async def test_tool_result_followup_chain_over_http(self):
        import httpx
        tools = [{'type': 'function', 'name': 'add', 'description': 'add',
                  'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']}}]
        bodies = []
        def backend(*args, **kwargs):
            bodies.append(args[0])
            if len(bodies) == 1:
                tc = {'index': 0, 'id': 'call_1', 'function': {'name': 'add', 'arguments': '{\"a\": 1}'}}
                yield {'choices': [{'delta': {'tool_calls': [tc]}, 'finish_reason': 'tool_calls'}]}
                yield {'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}}
            else:
                yield {'choices': [{'delta': {'content': 'The sum is 2.'}}]}
                yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
                yield {'choices': [], 'usage': {'prompt_tokens': 9, 'completion_tokens': 5, 'total_tokens': 14}}
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \\
                patch.object(script.shared.args, 'api_key', ''), \\
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                first = await client.post('/v1/responses', json={'input': 'Use the tool', 'tools': tools, 'stream': True})
                events = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith('data: ')]
                response = events[-1]['response']
                # Codex follows up by referencing the stored response and
                # appending only the tool result item.
                followup = await client.post('/v1/responses', json={
                    'previous_response_id': response['id'],
                    'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': '2'}],
                    'tools': tools, 'stream': True})
                followup_events = [json.loads(line[6:]) for line in followup.text.splitlines() if line.startswith('data: ')]
        self.assertEqual(events[-1]['type'], 'response.completed')
        self.assertEqual(response['output'][0]['type'], 'function_call')
        self.assertEqual(response['output'][0]['call_id'], 'call_1')
        self.assertEqual(followup_events[-1]['type'], 'response.completed')
        self.assertEqual(followup_events[-1]['response']['output'][0]['content'][0]['text'], 'The sum is 2.')
        # The stored history replayed the tool call and the new tool result.
        self.assertEqual(bodies[1]['messages'][-2]['role'], 'assistant')
        self.assertEqual(bodies[1]['messages'][-2]['tool_calls'][0]['id'], 'call_1')
        self.assertEqual(bodies[1]['messages'][-1]['role'], 'tool')
        self.assertEqual(bodies[1]['messages'][-1]['tool_call_id'], 'call_1')
        self.assertEqual(bodies[1]['messages'][-1]['content'], '2')

    async def test_queued_stream_announces_wait_and_runs_after_admission(self):
        import httpx
        def backend(*args, **kwargs):
            if args[0]['messages'][-1]['content'] == 'first':
                time.sleep(0.3)
                yield {'choices': [{'delta': {'content': 'one'}}]}
            else:
                yield {'choices': [{'delta': {'content': 'two'}}]}
            yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
            yield {'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \\
                patch.object(script.shared.args, 'api_key', ''), \\
                patch.object(script, 'RESPONSES_QUEUE_PING_SECONDS', 0.05), \\
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                first = await client.post('/v1/responses', json={'input': 'first', 'stream': True})
                second = await client.post('/v1/responses', json={'input': 'second', 'stream': True})
                first_events = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith('data: ')]
                second_events = [json.loads(line[6:]) for line in second.text.splitlines() if line.startswith('data: ')]
        self.assertEqual(first_events[-1]['type'], 'response.completed')
        self.assertEqual(second_events[0]['type'], 'response.created')
        self.assertIn('response.queued', [e['type'] for e in second_events])
        self.assertEqual(second_events[-1]['type'], 'response.completed')
        self.assertEqual(second_events[-1]['response']['output'][0]['content'][0]['text'], 'two')
        # Sequence numbers stay contiguous across queued announcements.
        self.assertEqual([e['sequence_number'] for e in second_events], list(range(len(second_events))))
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('tests added')
