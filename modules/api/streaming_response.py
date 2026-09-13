"""Signal cancellation before SSE waits for an in-flight synchronous generation step."""

from sse_starlette import EventSourceResponse


class GenerationEventSourceResponse(EventSourceResponse):
    def __init__(self, content, stop_event, **kwargs):
        self.stop_event = stop_event
        headers = dict(kwargs.pop('headers', {}) or {})
        headers.setdefault('Cache-Control', 'no-cache, no-store, must-revalidate')
        headers.setdefault('Connection', 'keep-alive')
        headers.setdefault('X-Accel-Buffering', 'no')
        kwargs['headers'] = headers
        # Codex may spend longer than a normal HTTP idle timeout in local
        # prefill or while executing a tool.  sse-starlette pings are actual
        # SSE comment frames and keep intermediaries from closing the stream.
        kwargs.setdefault('ping', 5)
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send):
        async def receive_with_cancellation():
            message = await receive()
            if message['type'] == 'http.disconnect':
                self.stop_event.set()
            return message

        try:
            await super().__call__(scope, receive_with_cancellation, send)
        finally:
            self.stop_event.set()
