"""Signal cancellation before SSE waits for an in-flight synchronous generation step."""

from sse_starlette import EventSourceResponse


class GenerationEventSourceResponse(EventSourceResponse):
    def __init__(self, content, stop_event, **kwargs):
        self.stop_event = stop_event
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
