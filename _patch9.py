path = r'F:\GitHub\textgen\tests\test_responses_api.py'
src = open(path, encoding='utf-8').read()

old = """        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \\
                patch.object(script.shared.args, 'api_key', ''), \\
                patch.object(script, 'RESPONSES_QUEUE_PING_SECONDS', 0.05), \\
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                first = await client.post('/v1/responses', json={'input': 'first', 'stream': True})
                second = await client.post('/v1/responses', json={'input': 'second', 'stream': True})
                first_events = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith('data: ')]
                second_events = [json.loads(line[6:]) for line in second.text.splitlines() if line.startswith('data: ')]
"""
new = """        with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \\
                patch.object(script.shared.args, 'api_key', ''), \\
                patch.object(script, 'RESPONSES_QUEUE_PING_SECONDS', 0.05), \\
                patch('sse_starlette.sse.AppStatus.should_exit_event', None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
                # Keep the first stream open and submit the second request
                # while the first generation is still running.
                async with client.stream('POST', '/v1/responses', json={'input': 'first', 'stream': True}) as first:
                    second = await client.post('/v1/responses', json={'input': 'second', 'stream': True})
                    second_lines = second.text.splitlines()
                first_lines = [line async for line in first.aiter_lines()]
                first_events = [json.loads(line[6:]) for line in first_lines if line.startswith('data: ')]
                second_events = [json.loads(line[6:]) for line in second_lines if line.startswith('data: ')]
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('test fixed')
