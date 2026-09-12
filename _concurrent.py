import asyncio, json, sys, threading, time
from unittest.mock import patch
import httpx

with patch('sys.argv', ['textgen-tests']):
    from modules.api import responses as api, script
    from modules.api.errors import InvalidRequestError

tools = [{'type': 'function', 'name': 'add', 'description': 'add',
          'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']}}]

def slow_tool_backend(*args, **kwargs):
    # First request: stream a long "thinking" prefix, then a tool call.
    for i in range(30):
        yield {'choices': [{'delta': {'content': 't' + str(i)}}]}
        time.sleep(0.05)
    tc = {'index': 0, 'id': 'call_1', 'function': {'name': 'add', 'arguments': '{"a": 1}'}}
    yield {'choices': [{'delta': {'tool_calls': [tc]}, 'finish_reason': 'tool_calls'}]}
    yield {'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 5, 'total_tokens': 10}}

calls = []
def backend(*args, **kwargs):
    calls.append((time.time(), args[0].get('messages', [{}])[-1]['content'][:40]))
    if len(calls) == 1:
        yield from slow_tool_backend(*args, **kwargs)
    else:
        yield {'choices': [{'delta': {'content': 'The answer is 2.'}}]}
        yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        yield {'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 4, 'total_tokens': 9}}

with patch.object(script.OAIcompletions, 'stream_chat_completions', side_effect=backend), \
     patch.object(script.shared.args, 'api_key', ''), \
     patch('sse_starlette.sse.AppStatus.should_exit_event', None):
    async def main():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=script.app), base_url='http://127.0.0.1') as client:
            t0 = time.time()
            r1 = await client.post('/v1/responses', json={'input': 'Use the tool', 'tools': tools, 'stream': True})
            print('r1 status', r1.status_code)
            # r1 stream is open; now submit the follow-up while r1 is still generating
            followup = {'input': [{'type': 'function_call', 'id': 'fc_x', 'call_id': 'call_1', 'name': 'add',
                                    'arguments': '{"a": 1}', 'status': 'completed'},
                                   {'type': 'function_call_output', 'call_id': 'call_1', 'output': '2'}],
                         'tools': tools, 'stream': True}
            r2 = await client.post('/v1/responses', json=followup)
            print('r2 accepted (status %s) at t=%.2fs' % (r2.status_code, time.time() - t0))
            # read r1 to completion
            events1 = [json.loads(l[6:]) for l in r1.text.splitlines() if l.startswith('data: ')]
            print('r1 events:', [e['type'] for e in events1])
            resp1 = events1[-1]['response']
            events2 = [json.loads(l[6:]) for l in r2.text.splitlines() if l.startswith('data: ')]
            print('r2 events:', [e['type'] for e in events2])
            print('r2 output:', events2[-1]['response']['output'])
            print('backend saw:', calls)
    asyncio.run(main())
