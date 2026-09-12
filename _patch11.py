path = r'F:\GitHub\textgen\modules\api\responses.py'
src = open(path, encoding='utf-8').read()

old = """    async def acquire(self):
        loop = asyncio.get_running_loop()
        ticket = loop.create_future()
        with self.lock:
            self.waiters.append(ticket)
            if len(self.waiters) == 1:
                ticket.set_result(True)
        return ticket

    def release(self, ticket):
"""
new = """    async def acquire(self):
        loop = asyncio.get_running_loop()
        ticket = loop.create_future()
        with self.lock:
            self.waiters.append(ticket)
            if len(self.waiters) == 1:
                ticket.set_result(True)
        return ticket

    def wait(self, ticket, stop_event, ping_interval=15.0):
        \"\"\"Async generator that waits for a ticket and yields one ping per
        interval while queued, so a streaming request can announce itself.
        Waiting uses no inference/threadpool worker. Cancellation checking
        must not cancel/re-enqueue the ticket, which would break FIFO.\"\"\"
        async def _wait():
            last_ping = time.monotonic()
            while not ticket.done() and not stop_event.is_set():
                await asyncio.wait([ticket], timeout=0.1)
                if not ticket.done() and time.monotonic() - last_ping >= ping_interval:
                    last_ping = time.monotonic()
                    yield 'queued'
        return _wait()

    def release(self, ticket):
"""
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('responses.py patched')
