"""Responses API compatibility over the local Chat Completions generator.

History is bounded, process-local CPU memory; it never manages the model KV cache.
"""
import copy
import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import ConfigDict, Field

from .errors import InvalidRequestError
from .typing import ChatCompletionRequest, GenerationOptions
from .responses_tools import ToolOutputError, custom_function, custom_input, strict_validator


class ResponsesRequest(GenerationOptions):
    model_config = ConfigDict(extra='forbid')
    input: str | list[dict] = Field(default_factory=list)
    model: str | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = True
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict = 'auto'
    parallel_tool_calls: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)
    text: dict | None = None
    reasoning: dict | None = None
    truncation: Literal['disabled', 'auto'] = 'disabled'
    background: bool = False
    include: list[str] = Field(default_factory=list)
    service_tier: str | None = None
    user: str | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal['in-memory', '24h'] | None = None
    stream_options: dict | None = None
    client_metadata: dict | None = None


def invalid(message, param='input', code=400):
    raise InvalidRequestError(message, param, code=code)


class ResponseStore:
    def __init__(self, max_entries=64, max_bytes=64 * 1024 * 1024, ttl=3600, clock=time.monotonic):
        self.max_entries, self.max_bytes, self.ttl, self.clock = max_entries, max_bytes, ttl, clock
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def _drop(self, key):
        _, size, _ = self.entries.pop(key)
        self.bytes -= size

    def _expire(self):
        now = self.clock()
        for key, (expires, _, _) in list(self.entries.items()):
            if expires <= now:
                self._drop(key)

    def put(self, response, history):
        # Stored snapshots are never mutated. Copy outside the lock so a large
        # context does not block unrelated lookups/deletions while copying.
        data = copy.deepcopy({'response': response, 'history': history})
        size = len(json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        if size > self.max_bytes:
            invalid('Response history exceeds the local storage limit; use store=false.', 'store', 413)
        with self.lock:
            self._expire()
            if response['id'] in self.entries:
                self._drop(response['id'])
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                self._drop(next(iter(self.entries)))
            self.entries[response['id']] = (self.clock() + self.ttl, size, data)
            self.bytes += size

    def get(self, response_id, *, field=None):
        with self.lock:
            self._expire()
            if response_id not in self.entries:
                invalid('Response not found, expired, evicted, or created with store=false.', 'response_id', 404)
            data = self.entries[response_id][2]
            if field is not None:
                data = data[field]
        # A reference keeps this immutable snapshot alive even if another
        # request deletes/evicts it after we release the lock.
        return copy.deepcopy(data)

    def delete(self, response_id):
        with self.lock:
            self._expire()
            if response_id not in self.entries:
                invalid('Response not found.', 'response_id', 404)
            self._drop(response_id)
        return {'id': response_id, 'object': 'response.deleted', 'deleted': True}


STORE = ResponseStore()


class GenerationQueue:
    """FIFO admission for Responses generation, without touching KV tensors."""
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


GENERATION_QUEUE = GenerationQueue()


class GenerationMetrics(dict):
    """One request's counters, retained when generation copies its state.

    Never share this object across requests. Its producer and consumer run in
    the same synchronous generator, including across streaming yields.
    """
    def __deepcopy__(self, memo):
        return self


def _string(item, key, param, allow_empty=False):
    value = item.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        invalid(f'{key} must be a string' + ('.' if allow_empty else ' and cannot be empty.'), param)
    return value


def _fields(item, allowed, param):
    unknown = set(item) - set(allowed.split())
    if unknown:
        invalid('Unsupported fields: ' + ', '.join(sorted(unknown)), param)


def function_definitions(tools):
    """Flatten namespaces for local chat templates; restore them on output."""
    for tool in tools:
        if tool.get('type') == 'namespace':
            _fields(tool, 'type name description tools', 'tools')
            namespace = _string(tool, 'name', 'tools')
            nested = tool.get('tools')
            if not isinstance(nested, list) or any(not isinstance(t, dict) or t.get('type') not in ('function', 'custom') for t in nested):
                invalid('Namespaces support function and custom tools only.', 'tools')
            for function in nested:
                name = _string(function, 'name', 'tools')
                yield {**function, 'name': namespace + '.' + name}
        else:
            yield tool


def output_call(request, call_id, name, arguments):
    if request.tool_choice == 'none':
        raise ToolOutputError('The model generated a tool call despite tool_choice=none.')
    item = {'type': 'function_call', 'id': 'fc_' + uuid.uuid4().hex,
            'call_id': call_id, 'name': name, 'arguments': arguments, 'status': 'completed'}
    for tool in function_definitions(request.tools):
        if tool.get('name') == name:
            if tool['type'] == 'custom':
                try:
                    item.update(type='custom_tool_call', input=custom_input(tool, arguments))
                except ValueError as exc:
                    raise ToolOutputError(str(exc)) from exc
                item.pop('arguments')
                item.pop('status')
            elif tool.get('strict'):
                try:
                    strict_validator(tool).validate(json.loads(arguments))
                except Exception as exc:
                    raise ToolOutputError('Generated function arguments failed strict schema validation for ' + name + '.') from exc
            break
    else:
        raise ToolOutputError('The model generated an undeclared tool: ' + str(name))
    for tool in request.tools:
        if tool.get('type') == 'namespace':
            for function in tool.get('tools', []):
                if name == tool['name'] + '.' + function['name']:
                    item.update(namespace=tool['name'], name=function['name'])
                    return item
    return item


def output_calls(request, calls):
    # Validate the whole batch before putting any executable item on the wire
    # or in a failed response's output array.
    if not request.parallel_tool_calls and len(calls) > 1:
        raise ToolOutputError('The model generated multiple calls with parallel_tool_calls=false; no calls were delivered.')
    ids = [call['id'] for call in calls]
    if any(not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ToolOutputError('The model generated missing or duplicate tool call IDs.')
    return [output_call(request, call['id'], call['name'], call['arguments']) for call in calls]


def _content(value, role, param):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        invalid('content must be a string or an array of content parts.', param)
    parts = []
    for part in value:
        if not isinstance(part, dict):
            invalid('Invalid content part.', param)
        kind = part.get('type')
        if kind in ('input_text', 'output_text'):
            _fields(part, 'type text annotations logprobs' if kind == 'output_text' else 'type text', param)
            if kind == 'output_text' and role != 'assistant':
                invalid('output_text is only valid for assistant messages.', param)
            parts.append({'type': 'text', 'text': _string(part, 'text', param, True)})
        elif kind == 'input_image' and role in ('user', 'tool'):
            _fields(part, 'type image_url file_id detail', param)
            if part.get('file_id'):
                invalid('Image file_id is unsupported; provide image_url or a data URL.', param)
            url = _string(part, 'image_url', param)
            if not url.startswith(('https://', 'http://', 'data:image/')):
                invalid('image_url must be an HTTP(S) or image data URL.', param)
            detail = part.get('detail', 'auto')
            if detail not in ('auto', 'low', 'high'):
                invalid('Unsupported image detail.', param)
            parts.append({'type': 'image_url', 'image_url': {'url': url, 'detail': detail}})
        else:
            invalid(f'Unsupported content type: {kind!r}.', param)
    if all(p['type'] == 'text' for p in parts):
        return '\n'.join(p['text'] for p in parts)
    return parts


def input_messages(items):
    if isinstance(items, str):
        return [{'role': 'user', 'content': items}]
    messages = []
    for index, item in enumerate(items):
        param = f'input.{index}'
        kind = item.get('type', 'message')
        if kind == 'message':
            _fields(item, 'type role content id status phase', param)
            role = item.get('role')
            if role not in ('user', 'assistant', 'system', 'developer'):
                invalid('Unsupported message role.', param)
            message = {'role': role, 'content': _content(item.get('content'), role, param)}
            if item.get('phase') is not None:
                if role != 'assistant' or item['phase'] not in ('commentary', 'final_answer'):
                    invalid('phase is only supported for assistant commentary/final_answer.', param)
                message['phase'] = item['phase']
            messages.append(message)
        elif kind in ('function_call', 'custom_tool_call'):
            _fields(item, 'type id call_id name arguments status namespace' if kind == 'function_call'
                    else 'type id call_id name input namespace', param)
            name = _string(item, 'name', param)
            if item.get('namespace') is not None:
                name = _string(item, 'namespace', param) + '.' + name
            call = {'id': _string(item, 'call_id', param), 'type': 'function',
                    'function': {'name': name,
                                 'arguments': (_string(item, 'arguments', param, True) if kind == 'function_call'
                                               else json.dumps({'input': _string(item, 'input', param, True)}, ensure_ascii=False))}}
            if messages and messages[-1]['role'] == 'assistant':
                messages[-1].setdefault('tool_calls', []).append(call)
            else:
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            _fields(item, 'type id call_id output status', param)
            messages.append({'role': 'tool', 'tool_call_id': _string(item, 'call_id', param),
                             'content': _content(item.get('output'), 'tool', param)})
        elif kind == 'reasoning':
            _fields(item, 'type id summary content encrypted_content status', param)
            if item.get('encrypted_content'):
                invalid('Encrypted reasoning cannot be used by a local model.', param)
            # Local reasoning items carry plaintext content, not a fabricated summary.
            content = item.get('content') or []
            if not isinstance(content, list):
                invalid('Reasoning content must be an array.', param)
            if item.get('summary') or any(not isinstance(p, dict) or p.get('type') != 'reasoning_text' for p in content):
                invalid('Only local reasoning_text items can be replayed.', param)
            thinking = ''.join(_string(p, 'text', param, True) for p in content)
            messages.append({'role': 'assistant', 'content': '', 'reasoning_content': thinking})
        else:
            invalid(f'Unsupported input item type: {kind!r}.', param)
    # Coalesce adjacent assistant reasoning/text/function items into one model turn.
    merged = []
    for message in messages:
        if (merged and message['role'] == merged[-1]['role'] == 'assistant'
                and not (message.get('phase') and merged[-1].get('phase') and message['phase'] != merged[-1]['phase'])):
            prior = merged[-1]
            prior['content'] += message['content']
            if message.get('phase'):
                prior['phase'] = message['phase']
            if message.get('reasoning_content'):
                prior['reasoning_content'] = prior.get('reasoning_content', '') + message['reasoning_content']
            if message.get('tool_calls'):
                prior.setdefault('tool_calls', []).extend(message['tool_calls'])
        else:
            merged.append(message)
    return merged


def prepare(request, store=STORE):
    # input/tools are normalized below; do not serialize the entire long
    # context a second time just to extract generation options.
    body = request.model_dump(include=set(GenerationOptions.model_fields) | {'temperature', 'top_p'})
    if request.prompt_cache_retention == '24h':
        invalid('The local KV cache cannot guarantee 24-hour retention; use in-memory.', 'prompt_cache_retention')
    if request.background:
        invalid('Background Responses are not implemented.', 'background')
    if set(request.include) - {'reasoning.encrypted_content'}:
        invalid('Additional include fields are not implemented.', 'include')
    if request.service_tier not in (None, 'auto', 'default'):
        invalid('Service tiers are not available for local generation.', 'service_tier')
    if request.stream_options and request.stream_options != {'include_obfuscation': False}:
        invalid('Only stream_options.include_obfuscation=false is supported.', 'stream_options')
    if request.text:
        _fields(request.text, 'format verbosity', 'text')
        if request.text.get('format', {'type': 'text'}) != {'type': 'text'}:
            invalid('Only text.format.type=text is supported; JSON Schema is unavailable.', 'text')
        if request.text.get('verbosity') not in (None, 'low', 'medium', 'high'):
            invalid('Unsupported text verbosity.', 'text.verbosity')
    if request.reasoning:
        if set(request.reasoning) - {'effort', 'summary'} or request.reasoning.get('summary') not in (None, 'none', 'auto', 'concise', 'detailed'):
            invalid('Unsupported reasoning options.', 'reasoning')
        effort = request.reasoning.get('effort')
        if effort is not None and effort not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh'):
            invalid('Unsupported reasoning effort.', 'reasoning.effort')
        if effort:
            body['reasoning_effort'] = effort
    if len(request.metadata) > 16 or any(len(k) > 64 or len(v) > 512 for k, v in request.metadata.items()):
        invalid('metadata allows 16 keys, key length <=64 and value length <=512.', 'metadata')
    if request.tool_choice not in ('auto', 'none'):
        invalid('Only auto/none tool_choice is supported; forced tool selection is not enforced by this backend.', 'tool_choice')
    tools = []
    names = set()
    for index, tool in enumerate(function_definitions(request.tools)):
        param = f'tools.{index}'
        if tool.get('type') == 'custom':
            _fields(tool, 'type name description format', param)
            _string(tool, 'name', param)
            if tool.get('description') is not None:
                _string(tool, 'description', param, True)
            try:
                tool = custom_function(tool)
            except Exception as exc:
                invalid('Invalid or unsupported custom tool format: ' + str(exc).splitlines()[0], param)
        if tool.get('type') != 'function':
            invalid('Only client-executed function tools are supported. In Codex set web_search="disabled" '
                    'to stop advertising OpenAI hosted web search.', param)
        if tool.get('strict') is True:
            try:
                strict_validator(tool)
            except Exception as exc:
                invalid('Invalid or unsupported strict tool schema: ' + str(exc).splitlines()[0], param)
        if tool.get('strict') is not None and type(tool['strict']) is not bool:
            invalid('strict must be a boolean or null.', param + '.strict')
        if tool.get('description') is not None and not isinstance(tool['description'], str):
            invalid('description must be a string.', param + '.description')
        if set(tool) - {'type', 'name', 'description', 'parameters', 'strict'}:
            invalid('Unsupported function tool fields.', param)
        name = _string(tool, 'name', param)
        if name in names:
            invalid('Duplicate function name.', param)
        names.add(name)
        parameters = tool.get('parameters') or {'type': 'object', 'properties': {}}
        if not isinstance(parameters, dict) or parameters.get('type') != 'object':
            invalid('Function parameters must be an object schema.', param)
        tools.append({'type': 'function', 'function': {'name': name, 'description': tool.get('description'), 'parameters': parameters}})

    history = []
    if request.previous_response_id:
        history = store.get(request.previous_response_id, field='history')
    history.extend(input_messages(request.input))
    if not history:
        invalid('input or previous_response_id must provide at least one message.')
    calls, results = set(), set()
    for message in history:
        if calls != results and message['role'] != 'tool':
            invalid('Return all pending function_call_output items before starting another message or tool turn.')
        for call in message.get('tool_calls', []):
            if call['id'] in calls:
                invalid('Duplicate function call_id.')
            calls.add(call['id'])
        if message['role'] == 'tool':
            cid = message['tool_call_id']
            if cid not in calls or cid in results:
                invalid('function_call_output must match a preceding unresolved call_id.')
            results.add(cid)
    if calls != results:
        invalid('Supply function_call_output for every pending function call before continuing.')
    messages = copy.deepcopy(history)
    if request.instructions is not None:
        messages.insert(0, {'role': 'system', 'content': request.instructions})
    if tools and request.tool_choice != 'none' and not request.parallel_tool_calls:
        messages.insert(0, {'role': 'system', 'content': 'Call at most one tool per response. Wait for its result before calling another tool.'})
    verbosity = (request.text or {}).get('verbosity')
    if verbosity in ('low', 'high'):
        messages.insert(0, {'role': 'system', 'content': (
            'Keep the final answer concise.' if verbosity == 'low' else 'Give a detailed final answer.')})
    params = {key: value for key, value in body.items() if key in GenerationOptions.model_fields}
    params.update(messages=messages, model=request.model, max_tokens=request.max_output_tokens,
                  stream=request.stream, stream_options={'include_usage': True}, tools=tools,
                  tool_choice=request.tool_choice)
    for key in ('temperature', 'top_p'):
        if body[key] is not None:
            params[key] = body[key]
    converted = ChatCompletionRequest(**params).model_dump()
    converted['_responses_no_truncation'] = request.truncation == 'disabled'
    converted['_responses_raise_errors'] = True
    converted['_responses_preserve_items'] = True
    converted['_responses_metrics'] = GenerationMetrics()
    return converted, history


def new_response(request, model):
    return {'id': 'resp_' + uuid.uuid4().hex, 'object': 'response', 'created_at': int(time.time()),
            'status': 'in_progress', 'error': None, 'incomplete_details': None, 'output': [],
            'model': model, 'instructions': request.instructions, 'max_output_tokens': request.max_output_tokens,
            'parallel_tool_calls': request.parallel_tool_calls, 'previous_response_id': request.previous_response_id,
            'reasoning': {'effort': (request.reasoning or {}).get('effort'), 'summary': None}, 'store': request.store,
            'temperature': request.temperature, 'top_p': request.top_p,
            'text': {'format': {'type': 'text'}, **({'verbosity': request.text['verbosity']} if request.text and 'verbosity' in request.text else {})},
            'tool_choice': request.tool_choice, 'tools': request.tools, 'truncation': request.truncation,
            'usage': None, 'metadata': request.metadata, 'user': request.user, 'background': False}


def usage(value):
    if not value:
        return None
    result = {'input_tokens': value.get('prompt_tokens', 0), 'output_tokens': value.get('completion_tokens', 0),
              'total_tokens': value.get('total_tokens', 0)}
    # Omit unknown breakdowns instead of fabricating cache hits or reasoning counts.
    if value.get('prompt_tokens_details') is not None:
        result['input_tokens_details'] = copy.deepcopy(value['prompt_tokens_details'])
    if value.get('completion_tokens_details') is not None:
        result['output_tokens_details'] = copy.deepcopy(value['completion_tokens_details'])
    return result


def message_item(text, status='completed'):
    return {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message', 'role': 'assistant', 'status': status,
            'content': [{'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}]}


def complete_response(response, finish_reason, token_usage):
    response['status'] = 'incomplete' if finish_reason == 'length' else 'completed'
    response['incomplete_details'] = {'reason': 'max_output_tokens'} if finish_reason == 'length' else None
    response['usage'] = usage(token_usage)


def save_response(request, response, history, store=STORE):
    if request.store:
        store.put(response, history + input_messages(response['output']))


def from_chat(request, chat, history, store=STORE):
    response = new_response(request, chat['model'])
    choice = chat['choices'][0]
    message = choice['message']
    if message.get('reasoning_content'):
        response['output'].append({'type': 'reasoning', 'id': 'rs_' + uuid.uuid4().hex, 'summary': [],
                                   'content': [{'type': 'reasoning_text', 'text': message['reasoning_content']}]})
    response['output'].extend(output_calls(request, [
        {'id': call['id'], **call['function']} for call in message.get('tool_calls', [])]))
    if message.get('content') or not response['output']:
        response['output'].append(message_item(message.get('content') or '', 'incomplete' if choice['finish_reason'] == 'length' else 'completed'))
    complete_response(response, choice['finish_reason'], chat.get('usage'))
    save_response(request, response, history, store)
    return response


class StreamConverter:
    def __init__(self, request, history, model, store=STORE):
        self.request, self.history, self.store = request, history, store
        self.response = new_response(request, model)
        self.sequence = 0
        self.message = None
        self.reasoning = None
        self.calls = OrderedDict()
        self.finish_reason = None
        self.token_usage = None
        self.buffer_text = bool(request.tools) and request.tool_choice != 'none'
        self.pending_text = ''

    def event(self, kind, **data):
        # json.dumps snapshots values immediately; a deep copy here only adds
        # work for complete output items and long terminal responses.
        event = {'type': kind, 'sequence_number': self.sequence, **data}
        self.sequence += 1
        return {'event': kind, 'data': json.dumps(event, ensure_ascii=False)}

    def start(self):
        return [self.event('response.created', response=self.response), self.event('response.in_progress', response=self.response)]

    def add_text(self, text):
        events = []
        if self.message is None:
            self.message = message_item('', 'in_progress')
            self.message_index = len(self.response['output'])
            self.response['output'].append(self.message)
            empty = {**self.message, 'content': []}
            events.append(self.event('response.output_item.added', output_index=self.message_index, item=empty))
            events.append(self.event('response.content_part.added', output_index=self.message_index,
                                     item_id=self.message['id'], content_index=0, part=self.message['content'][0]))
        self.message['content'][0]['text'] += text
        if text:
            events.append(self.event('response.output_text.delta', output_index=self.message_index,
                                     item_id=self.message['id'], content_index=0, delta=text, logprobs=[]))
        return events

    def process(self, chunk):
        events = []
        if chunk.get('model'):
            self.response['model'] = chunk['model']
        if chunk.get('usage'):
            self.token_usage = chunk['usage']
        for choice in chunk.get('choices', []):
            if choice.get('finish_reason'):
                self.finish_reason = choice['finish_reason']
            delta = choice.get('delta', {})
            if delta.get('reasoning_content'):
                if self.reasoning is None:
                    self.reasoning = {'type': 'reasoning', 'id': 'rs_' + uuid.uuid4().hex, 'summary': [],
                                      'content': [{'type': 'reasoning_text', 'text': ''}]}
                    self.reasoning_index = len(self.response['output'])
                    self.response['output'].append(self.reasoning)
                    events.append(self.event('response.output_item.added', output_index=self.reasoning_index, item={**self.reasoning, 'content': []}))
                    events.append(self.event('response.content_part.added', output_index=self.reasoning_index,
                                             item_id=self.reasoning['id'], content_index=0, part=self.reasoning['content'][0]))
                self.reasoning['content'][0]['text'] += delta['reasoning_content']
                events.append(self.event('response.reasoning_text.delta', output_index=self.reasoning_index,
                                         item_id=self.reasoning['id'], content_index=0, delta=delta['reasoning_content']))
            if delta.get('content'):
                if self.buffer_text:
                    self.pending_text += delta['content']
                else:
                    events.extend(self.add_text(delta['content']))
            for call in delta.get('tool_calls', []):
                index = call.get('index', 0)
                if index not in self.calls:
                    self.calls[index] = {'id': '', 'name': '', 'arguments': ''}
                assembled = self.calls[index]
                if call.get('id'):
                    assembled['id'] = call['id']
                function = call.get('function', {})
                assembled['name'] += function.get('name', '')
                assembled['arguments'] += function.get('arguments', '')
        return events

    def finish(self):
        if self.finish_reason is None:
            raise RuntimeError('Chat stream ended without a finish reason')
        events = []
        if self.calls:
            # Calls are assembled first; backend may only identify them at the end.
            for item in output_calls(self.request, list(self.calls.values())):
                index = len(self.response['output'])
                self.response['output'].append(item)
                if item['type'] == 'custom_tool_call':
                    events.append(self.event('response.output_item.added', output_index=index, item={**item, 'input': ''}))
                    events.append(self.event('response.custom_tool_call_input.delta', output_index=index, item_id=item['id'], delta=item['input']))
                    events.append(self.event('response.custom_tool_call_input.done', output_index=index, item_id=item['id'], input=item['input']))
                else:
                    events.append(self.event('response.output_item.added', output_index=index,
                                             item={**item, 'arguments': '', 'status': 'in_progress'}))
                    events.append(self.event('response.function_call_arguments.delta', output_index=index, item_id=item['id'], delta=item['arguments']))
                    events.append(self.event('response.function_call_arguments.done', output_index=index, item_id=item['id'], arguments=item['arguments'], name=item['name']))
                events.append(self.event('response.output_item.done', output_index=index, item=item))
        elif self.pending_text or not self.response['output']:
            events.extend(self.add_text(self.pending_text))
        if self.reasoning is not None:
            part = self.reasoning['content'][0]
            events.append(self.event('response.reasoning_text.done', output_index=self.reasoning_index,
                                     item_id=self.reasoning['id'], content_index=0, text=part['text']))
            events.append(self.event('response.content_part.done', output_index=self.reasoning_index,
                                     item_id=self.reasoning['id'], content_index=0, part=part))
            events.append(self.event('response.output_item.done', output_index=self.reasoning_index, item=self.reasoning))
        if self.message is not None:
            part = self.message['content'][0]
            self.message['status'] = 'incomplete' if self.finish_reason == 'length' else 'completed'
            events.append(self.event('response.output_text.done', output_index=self.message_index,
                                     item_id=self.message['id'], content_index=0, text=part['text'], logprobs=[]))
            events.append(self.event('response.content_part.done', output_index=self.message_index,
                                     item_id=self.message['id'], content_index=0, part=part))
            events.append(self.event('response.output_item.done', output_index=self.message_index, item=self.message))
        complete_response(self.response, self.finish_reason, self.token_usage)
        save_response(self.request, self.response, self.history, self.store)
        events.append(self.event('response.' + self.response['status'], response=self.response))
        return events

    def failed(self, message, code='server_error'):
        self.response['status'] = 'failed'
        self.response['error'] = {'code': code, 'message': message}
        return self.event('response.failed', response=self.response)
