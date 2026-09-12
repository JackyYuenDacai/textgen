path = r'F:\GitHub\textgen\modules\api\script.py'
src = open(path, encoding='utf-8').read()

old = "import json\nimport logging\nimport os\nimport socket\nimport threading\nimport traceback\n"
new = "import json\nimport logging\nimport os\nimport socket\nimport threading\nimport time\nimport traceback\n"
assert src.count(old) == 1
src = src.replace(old, new)

old = """@app.post('/v1/responses', dependencies=check_key)
async def openai_responses(request: Request, request_data: Responses.ResponsesRequest):
    # Validate and resolve history before returning HTTP 200 / starting SSE.
    converted, history = await asyncio.to_thread(Responses.prepare, request_data)
    stop_event = threading.Event()
    if request_data.stream:
        async def generator():
            converter = Responses.StreamConverter(request_data, history, shared.model_name or 'unknown')
            async with Responses.GENERATION_QUEUE.slot(stop_event) as admitted:
                for event in converter.start():
                    yield event
                if not admitted:
                    # A client waiting in the queue must not hang on an empty
                    # stream when the request is never generated.
                    yield converter.failed('Server is busy; the request was not generated.', 'server_error')
                    return
                response = OAIcompletions.stream_chat_completions(converted, stop_event=stop_event)
                try:
                    async for chunk in iterate_in_threadpool(response):
                        if stop_event.is_set():
                            return
                        for event in converter.process(chunk):
                            yield event
                    if not stop_event.is_set():
                        # Grammar/schema validation and history snapshots can
                        # be expensive; keep queue pings and disconnects live.
                        for event in await asyncio.to_thread(converter.finish):
                            yield event
                except OpenAIError as error:
                    yield converter.failed(error.message, 'invalid_prompt' if error.code < 500 else 'server_error')
                except Responses.ToolOutputError as error:
                    yield converter.failed(str(error), 'model_output_invalid')
                except Exception:
                    logger.exception('Responses generation failed')
                    yield converter.failed('Local generation failed; check the server log.')
                finally:
                    stop_event.set()
                    response.close()
        return GenerationEventSourceResponse(generator(), stop_event, sep='\\n')
"""
new = """# How often a queued streaming request announces itself while waiting for
# the single local generation slot.
RESPONSES_QUEUE_PING_SECONDS = 15.0


@app.post('/v1/responses', dependencies=check_key)
async def openai_responses(request: Request, request_data: Responses.ResponsesRequest):
    # Validate and resolve history before returning HTTP 200 / starting SSE.
    converted, history = await asyncio.to_thread(Responses.prepare, request_data)
    stop_event = threading.Event()
    if request_data.stream:
        async def generator():
            converter = Responses.StreamConverter(request_data, history, shared.model_name or 'unknown')
            for event in converter.created():
                yield event
            ticket = await Responses.GENERATION_QUEUE.acquire()
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
                    # A client waiting in the queue must not hang on an empty
                    # stream when the request is never generated.
                    yield converter.failed('Server is busy; the request was not generated.', 'server_error')
                    return
                for event in converter.admitted():
                    yield event
                response = OAIcompletions.stream_chat_completions(converted, stop_event=stop_event)
                try:
                    async for chunk in iterate_in_threadpool(response):
                        if stop_event.is_set():
                            return
                        for event in converter.process(chunk):
                            yield event
                    if not stop_event.is_set():
                        # Grammar/schema validation and history snapshots can
                        # be expensive; keep queue pings and disconnects live.
                        for event in await asyncio.to_thread(converter.finish):
                            yield event
                except OpenAIError as error:
                    yield converter.failed(error.message, 'invalid_prompt' if error.code < 500 else 'server_error')
                except Responses.ToolOutputError as error:
                    yield converter.failed(str(error), 'model_output_invalid')
                except Exception:
                    logger.exception('Responses generation failed')
                    yield converter.failed('Local generation failed; check the server log.')
                finally:
                    stop_event.set()
                    response.close()
            finally:
                Responses.GENERATION_QUEUE.release(ticket)
        return GenerationEventSourceResponse(generator(), stop_event, sep='\\n')
"""
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('script.py patched')
