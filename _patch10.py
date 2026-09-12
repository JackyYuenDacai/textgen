path = r'F:\GitHub\textgen\tests\test_responses_api.py'
src = open(path, encoding='utf-8').read()

old = """                # Keep the first stream open and submit the second request
                # while the first generation is still running.
                async with client.stream('POST', '/v1/responses', json={'input': 'first', 'stream': True}) as first:
                    second = await client.post('/v1/responses', json={'input': 'second', 'stream': True})
                    second_lines = second.text.splitlines()
                first_lines = [line async for line in first.aiter_lines()]
                first_events = [json.loads(line[6:]) for line in first_lines if line.startswith('data: ')]
                second_events = [json.loads(line[6:]) for line in second_lines if line.startswith('data: ')]
"""
new = """                # Keep the first stream open and submit the second request
                # while the first generation is still running.
                async with client.stream('POST', '/v1/responses', json={'input': 'first', 'stream': True}) as first:
                    async def drain():
                        return [line async for line in first.aiter_lines()]
                    first_task = asyncio.create_task(drain())
                    second = await client.post('/v1/responses', json={'input': 'second', 'stream': True})
                    second_lines = second.text.splitlines()
                    first_lines = await first_task
                first_events = [json.loads(line[6:]) for line in first_lines if line.startswith('data: ')]
                second_events = [json.loads(line[6:]) for line in second_lines if line.startswith('data: ')]
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('test fixed')
