import sys, json
sys.path.insert(0, '.')
from modules.api import responses as R

tools = [{'type': 'function', 'name': 'add', 'description': 'add numbers',
          'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}}]

# --- Turn 1: streaming tool call (as Codex sends it) ---
req1 = R.ResponsesRequest(input='What is 1+2? Use the tool.', tools=tools, stream=True, store=True,
                          reasoning={'effort': 'medium', 'summary': 'auto'},
                          text={'format': {'type': 'text'}}, truncation='auto')
body1, history1 = R.prepare(req1)
conv = R.StreamConverter(req1, history1, 'local-model')
events = conv.start()
events += conv.process({'model': 'local-model', 'choices': [{'delta': {'content': ''}}]})
tc = {'index': 0, 'id': 'call_abc', 'function': {'name': 'add', 'arguments': '{"a": 1, "b": 2}'}}
events += conv.process({'model': 'local-model', 'choices': [{'delta': {'tool_calls': [tc]}, 'finish_reason': 'tool_calls'}]})
events += conv.process({'model': 'local-model', 'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
events += conv.finish()
final = json.loads(events[-1]['data'])
final = final.get('response', final)
print('turn1 status:', final['status'], 'finish event:', events[-1]['event'])
print('turn1 output:', json.dumps(final['output'], indent=1)[:600])
resp_id = final['id']

# --- Turn 2: Codex-style follow-up using previous_response_id ---
out_item = {'type': 'function_call_output', 'call_id': 'call_abc', 'output': '3'}
req2 = R.ResponsesRequest(previous_response_id=resp_id, input=[out_item], tools=tools, stream=True, store=True)
try:
    body2, history2 = R.prepare(req2)
    print('turn2 (previous_response_id) OK; last messages:')
    for m in body2['messages'][-4:]:
        print('  ', json.dumps(m)[:200])
except Exception as e:
    print('turn2 (previous_response_id) FAILED:', type(e).__name__, e)

# --- Turn 2b: Codex-style follow-up with full item list (no previous_response_id) ---
echo = final['output']
req3 = R.ResponsesRequest(input=echo + [out_item], tools=tools, stream=True, store=True)
try:
    body3, history3 = R.prepare(req3)
    print('turn2b (full items) OK; last messages:')
    for m in body3['messages'][-4:]:
        print('  ', json.dumps(m)[:200])
except Exception as e:
    print('turn2b (full items) FAILED:', type(e).__name__, e)

