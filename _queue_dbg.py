import asyncio, json, sys, threading, time
from unittest.mock import patch
import httpx

with patch('sys.argv', ['textgen-tests']):
    from modules.api import responses as api, script
    from modules.api.errors import InvalidRequestError

def backend(*args, **kwargs):
    which = args[0]['messages'][-1]['content']
    print(f'[{time.monotonic():.3f}] backend start: {which}', flush=True)
    if which == 'first':
        time.sleep(0.3)
        yield {'choices': [{'delta': {'content': 'one'}}]}
    else:
        yield {'choices': [{'delta': {'content': 'two'}}]}
    yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
    yield {'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
    print(f'[{time.monotonic():.3f}] backend end: {which}', flush=True)

with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
     patch.object(script.shared.args, 'api_key', ''), \
     patch.object(script, 'RESPONSES_QUEUE_PING_SECONDS', 0.05), \
     patch('sse_starlette.sse.AppStatus.should_exit_event', None):
    async def main():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
            async with client.stream('POST', '/v1/responses', json={'input': 'first', 'stream': True}) as first:
                print(f'[{time.monotonic():.3f}] first stream open', flush=True)
                async def drain():
                    lines = [line async for line in first.aiter_lines()]
                    print(f'[{time.monotonic():.3f}] first drained', flush=True)
                    return lines
                first_task = asyncio.create_task(drain())
                print(f'[{time.monotonic():.3f}] sending second', flush=True)
                second = await client.post('/v1/responses', json={'input': 'second', 'stream': True})
                print(f'[{time.monotonic():.3f}] second done', flush=True)
                first_lines = await first_task
        print('FIRST:', [json.loads(l[6:])['type'] for l in first_lines if l.startswith('data: ')])
        print('SECOND:', [json.loads(l[6:])['type'] for l in second.text.splitlines() if l.startswith('data: ')])
    asyncio.run(main())
