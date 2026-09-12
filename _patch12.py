path = r'F:\GitHub\textgen\modules\api\script.py'
src = open(path, encoding='utf-8').read()

old = """            ticket = await Responses.GENERATION_QUEUE.acquire()
            try:
                # The local model serves one generation at a time. A queued
                # request keeps its stream alive and announces itself instead
                # of hanging, so clients can follow up (for example with a
                # tool result) as soon as it is admitted.
                last_ping = time.monotonic()
                while not ticket.done() and not stop_event.is_set():
                    await asyncio.wait([ticket], timeout=0.1)
                    if not ticket.done() and time.monotonic() - last_ping >= RESPONSES_QUEUE_PING_SECONDS:
                        last_ping = time.monotonic()
                        for event in converter.queued():
                            yield event
                if stop_event.is_set():
"""
new = """            ticket = await Responses.GENERATION_QUEUE.acquire()
            try:
                # The local model serves one generation at a time. A queued
                # request keeps its stream alive and announces itself instead
                # of hanging, so clients can follow up (for example with a
                # tool result) as soon as it is admitted.
                async for _ in Responses.GENERATION_QUEUE.wait(ticket, stop_event, RESPONSES_QUEUE_PING_SECONDS):
                    for event in converter.queued():
                        yield event
                if stop_event.is_set():
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('script.py patched')
