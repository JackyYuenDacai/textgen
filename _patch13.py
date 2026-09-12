path = r'F:\GitHub\textgen\tests\test_responses_api.py'
src = open(path, encoding='utf-8').read()

old = """    async def test_queued_stream_announces_wait_and_runs_after_admission(self):
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
                # Keep the first stream open and submit the second request
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
        self.assertEqual(first_events[-1]['type'], 'response.completed')
        self.assertEqual(second_events[0]['type'], 'response.created')
        self.assertIn('response.queued', [e['type'] for e in second_events])
        self.assertEqual(second_events[-1]['type'], 'response.completed')
        self.assertEqual(second_events[-1]['response']['output'][0]['content'][0]['text'], 'two')
        # Sequence numbers stay contiguous across queued announcements.
        self.assertEqual([e['sequence_number'] for e in second_events], list(range(len(second_events))))
"""
new = """    async def test_queued_wait_pings_until_admission_and_releases_fifo(self):
        queue = api.GenerationQueue()
        stop = threading.Event()
        first = await queue.acquire()
        second = await queue.acquire()
        pings = []
        async def wait_second():
            async for ping in queue.wait(second, stop, 0.05):
                pings.append(ping)
        waiter = asyncio.create_task(wait_second())
        await asyncio.sleep(0.25)
        self.assertGreaterEqual(len(pings), 2)
        queue.release(first)
        await asyncio.wait_for(waiter, 2)
        self.assertTrue(all(ping == 'queued' for ping in pings))
        self.assertEqual(len(queue.waiters), 0)
        # A request that stops while queued is denied without breaking FIFO.
        denied = threading.Event()
        third = await queue.acquire()
        async def wait_third():
            async for _ in queue.wait(third, denied, 0.05):
                pass
        third_waiter = asyncio.create_task(wait_third())
        await asyncio.sleep(0.1)
        denied.set()
        await asyncio.wait_for(third_waiter, 2)
        queue.release(third)
        self.assertEqual(len(queue.waiters), 0)
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('test replaced')
