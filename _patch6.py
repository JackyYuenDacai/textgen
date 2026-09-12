path = r'F:\GitHub\textgen\modules\api\responses.py'
src = open(path, encoding='utf-8').read()

old = """class GenerationQueue:
    \"\"\"FIFO admission for Responses generation, without touching KV tensors.\"\"\"
    def __init__(self):
        self.lock = threading.Lock()
        self.waiters = deque()

    @asynccontextmanager
    async def slot(self, stop_event):
        loop = asyncio.get_running_loop()
        ticket = loop.create_future()
        with self.lock:
            self.waiters.append(ticket)
            if len(self.waiters) == 1:
                ticket.set_result(True)
        try:
            # Waiting uses no inference/threadpool worker. Cancellation checking
            # must not cancel/re-enqueue the ticket, which would break FIFO.
            while not ticket.done() and not stop_event.is_set():
                await asyncio.wait([ticket], timeout=0.1)
            yield not stop_event.is_set()
        finally:
            with self.lock:
                self.waiters.remove(ticket)
                if self.waiters and not self.waiters[0].done():
                    next_ticket = self.waiters[0]
                    next_ticket.get_loop().call_soon_threadsafe(self._wake, next_ticket)

    @staticmethod
    def _wake(ticket):
        if not ticket.done():
            ticket.set_result(True)
"""
new = """class GenerationQueue:
    \"\"\"FIFO admission for Responses generation, without touching KV tensors.\"\"\"
    def __init__(self):
        self.lock = threading.Lock()
        self.waiters = deque()

    async def acquire(self):
        loop = asyncio.get_running_loop()
        ticket = loop.create_future()
        with self.lock:
            self.waiters.append(ticket)
            if len(self.waiters) == 1:
                ticket.set_result(True)
        return ticket

    def release(self, ticket):
        with self.lock:
            self.waiters.remove(ticket)
            if self.waiters and not self.waiters[0].done():
                next_ticket = self.waiters[0]
                next_ticket.get_loop().call_soon_threadsafe(self._wake, next_ticket)

    @asynccontextmanager
    async def slot(self, stop_event):
        ticket = await self.acquire()
        try:
            # Waiting uses no inference/threadpool worker. Cancellation checking
            # must not cancel/re-enqueue the ticket, which would break FIFO.
            while not ticket.done() and not stop_event.is_set():
                await asyncio.wait([ticket], timeout=0.1)
            yield not stop_event.is_set()
        finally:
            self.release(ticket)

    @staticmethod
    def _wake(ticket):
        if not ticket.done():
            ticket.set_result(True)
"""
assert src.count(old) == 1
src = src.replace(old, new)

old = """    def start(self):
        return [self.event('response.created', response=self.response), self.event('response.in_progress', response=self.response)]
"""
new = """    def created(self):
        return [self.event('response.created', response=self.response)]

    def queued(self):
        # Announced while the request waits for the single local generation
        # slot, so clients stay informed instead of seeing a silent stream.
        return [self.event('response.queued', response=self.response)]

    def admitted(self):
        return [self.event('response.in_progress', response=self.response)]

    def start(self):
        return self.created() + self.admitted()
"""
assert src.count(old) == 1
src = src.replace(old, new)

open(path, 'w', encoding='utf-8', newline='').write(src)
print('responses.py patched')
