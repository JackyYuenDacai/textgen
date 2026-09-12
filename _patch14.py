path = r'F:\GitHub\textgen\tests\test_responses_api.py'
src = open(path, encoding='utf-8').read()

old = """        queue.release(first)
        await asyncio.wait_for(waiter, 2)
        self.assertTrue(all(ping == 'queued' for ping in pings))
        self.assertEqual(len(queue.waiters), 0)
"""
new = """        queue.release(first)
        await asyncio.wait_for(waiter, 2)
        queue.release(second)
        self.assertTrue(all(ping == 'queued' for ping in pings))
        self.assertEqual(len(queue.waiters), 0)
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('fixed')
