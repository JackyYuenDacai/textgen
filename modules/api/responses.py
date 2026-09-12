"""Responses API serializer over native semantic generation events.

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
from .typing import GenerationRequestOptions, GenerationOptions
from .canonical_items import ConversationState, render_items as input_messages
from .generation_events import EventBatch, StartedEvent, TextDelta, ReasoningDelta, ToolCallDelta, DoneEvent, UsageEvent
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

    def put(self, response, history, *, input_count=None):
        # Stored snapshots are never mutated. Copy outside the lock so a large
        # context does not block unrelated lookups/deletions while copying.
        data = copy.deepcopy({'response': response, 'history': history,
                              'input_count': len(history) if input_count is None else input_count})
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

    def _snapshot(self, response_id):
        with self.lock:
            self._expire()
            if response_id not in self.entries:
                invalid('Response not found, expired, evicted, or created with store=false.', 'response_id', 404)
            # Read/chain access marks the entry most-recently-used so an
            # active conversation survives FIFO-style eviction by idle ones.
            self.entries.move_to_end(response_id)
            return self.entries[response_id][2]

    def get(self, response_id, *, field=None):
        data = self._snapshot(response_id)
        if field is not None:
            data = data[field]
        # A reference keeps this immutable snapshot alive even if another
        # request deletes/evicts it after we release the lock.
        return copy.deepcopy(data)

    def input_items(self, response_id, *, after=None, limit=20, order='desc', include=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            invalid('limit must be between 1 and 100.', 'limit')
        if order not in ('asc', 'desc'):
            invalid('order must be asc or desc.', 'order')
        if set(include or []) - {'message.input_image.image_url', 'message.output_text.logprobs', 'reasoning.encrypted_content'}:
            invalid('Additional include fields are not implemented.', 'include')
        snapshot = self._snapshot(response_id)
        items, count = snapshot['history'], snapshot['input_count']
        indices = range(count) if order == 'asc' else range(count - 1, -1, -1)
        if after is not None:
            position = next((i for i, index in enumerate(indices) if items[index]['id'] == after), None)
            if position is None:
                invalid('after must identify an input item in this response.', 'after')
            indices = indices[position + 1:]
        # Immutable snapshots remain alive across deletion/eviction. Copy only
        # this page, outside the lock, never the whole long-context history.
        page = copy.deepcopy([items[index] for index in indices[:limit]])
        return {'object': 'list', 'data': page, 'has_more': len(indices) > limit,
                'first_id': page[0]['id'] if page else None,
                'last_id': page[-1]['id'] if page else None}

    def delete(self, response_id):
        with self.lock:
            self._expire()
            if response_id not in self.entries:
                invalid('Response not found.', 'response_id', 404)
            self._drop(response_id)
        return {'id': response_id, 'object': 'response.deleted', 'deleted': True}


STORE = ResponseStore()


def canonical_input_items(value, previous=()):
    """Preserve wire items before lossy conversion to backend chat messages."""
    raw = [{'role': 'user', 'content': value}] if isinstance(value, str) else value
    items = []
    ids = {item['id'] for item in previous}
    for index, original in enumerate(raw):
        item = copy.deepcopy(original)
        kind = item.setdefault('type', 'message')
        if item.get('id') is None:
            prefix = {'message': 'msg_', 'reasoning': 'rs_', 'function_call': 'fc_',
                      'custom_tool_call': 'ctc_'}.get(kind, 'item_')
            item['id'] = prefix + uuid.uuid4().hex
        identity = _string(item, 'id', f'input.{index}.id')
        if identity in ids:
            invalid('Duplicate input item id.', f'input.{index}.id')
        ids.add(identity)
        if kind == 'message':
            if isinstance(item['content'], str):
                part_type = 'output_text' if item['role'] == 'assistant' else 'input_text'
                item['content'] = [{'type': part_type, 'text': item['content']}]
            for part in item['content']:
                if item['role'] == 'assistant' and part['type'] == 'input_text':
                    part['type'] = 'output_text'
                if part['type'] == 'output_text':
                    part.setdefault('annotations', [])
                    part.setdefault('logprobs', [])
                elif part['type'] == 'input_image':
                    part.setdefault('detail', 'auto')
            item.setdefault('status', 'completed')
        elif kind == 'reasoning':
            item.setdefault('summary', [])
        items.append(item)
    return items


class GenerationQueue:
    """FIFO admission for Responses generation, without touching KV tensors."""
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

    def wait(self, ticket, stop_event, ping_interval=15.0):
        """Async generator that waits for a ticket and yields one ping per
        interval while queued, so a streaming request can announce itself.
        Waiting uses no inference/threadpool worker. Cancellation checking
        must not cancel/re-enqueue the ticket, which would break FIFO."""
        async def _wait():
            last_ping = time.monotonic()
            while not ticket.done() and not stop_event.is_set():
                await asyncio.wait([ticket], timeout=0.1)
                if not ticket.done() and time.monotonic() - last_ping >= ping_interval:
                    last_ping = time.monotonic()
                    yield 'queued'
        return _wait()

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
        if tool.get('type') in ('web_search', 'web_search_preview', 'file_search', 'computer_use_preview'):
            invalid('Hosted tools are not supported by the local Responses backend.', param)
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
    # Validate before canonicalizing. Keep the public item types and IDs for
    # retrieval; only the generation path needs flattened chat messages.
    messages = input_messages(history) + input_messages(request.input)
    if not messages:
        invalid('input or previous_response_id must provide at least one message.')
    calls, results = set(), set()
    for message in messages:
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
    history.extend(canonical_input_items(request.input, history))
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
    converted = GenerationRequestOptions(**params).model_dump()
    # Transitional wire view retained for Chat/legacy callers; the canonical
    # item graph above is the source of truth for Responses state.
    converted['messages'] = copy.deepcopy(messages)
    prefix = messages[:len(messages) - len(input_messages(history))]
    canonical = ConversationState.from_items([
        *({'type': 'message', **message} for message in prefix), *history])
    # IDs are response metadata, never part of the prompt graph for new input.
    for item in canonical.items:
        item.data.pop('id', None)
    converted['items'] = canonical.items
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
        store.put(response, history + response['output'], input_count=len(history))


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
        # NativeGeneration emits tool calls as separate semantic events, so
        # buffering all text whenever tools are present makes the final answer
        # appear only at completion. Keep text streaming; the legacy adapter
        # still has its own tool-call filtering in process().
        # Legacy chat-chunk providers may emit tool markup as ordinary text
        # before the structured tool call arrives. Buffer that text so it is
        # never exposed as answer content. Native EventBatch generation does
        # not use this branch and remains fully streaming.
        self.buffer_text = bool(request.tools) and request.tool_choice != 'none'
        self.pending_text = ''

    def event(self, kind, **data):
        # json.dumps snapshots values immediately; a deep copy here only adds
        # work for complete output items and long terminal responses.
        event = {'type': kind, 'sequence_number': self.sequence, **data}
        self.sequence += 1
        return {'event': kind, 'data': json.dumps(event, ensure_ascii=False)}

    def created(self):
        return [self.event('response.created', response=self.response)]

    def queued(self):
        # Announced while the request waits for the single local generation
        # slot, so clients stay informed instead of seeing a silent stream.
        return [self.event('response.queued', response=self.response)]

    def admitted(self):
        return [self.event('response.in_progress', response=self.response)]

    def start(self):
        return self.created() + self.admitted()

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
        # Native generation path: adapt semantic events directly at the
        # Responses boundary. The legacy Chat-chunk branch below remains for
        # third-party backends and backwards compatibility.
        if isinstance(chunk, EventBatch):
            return self.process_events(chunk)
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
                # Responses API names reasoning stream deltas as summary text
                # events.  Keep the wire event aligned with the public API so
                # clients can route reasoning separately from answer text.
                events.append(self.event('response.reasoning_summary_text.delta', output_index=self.reasoning_index,
                                         item_id=self.reasoning['id'], content_index=0, delta=delta['reasoning_content']))
            if delta.get('content'):
                if self.buffer_text:
                    self.pending_text += delta['content']
                else:
                    events.extend(self.add_text(delta['content']))
            for call in delta.get('tool_calls', []):
                index = call.get('index', 0)
                is_new = index not in self.calls
                if is_new:
                    # Emit the item as soon as the provider opens a tool call.
                    # Waiting until the terminal chunk makes a Responses
                    # consumer blind during long argument streams.
                    function = call.get('function', {})
                    # The item ID is server-generated and must stay stable
                    # across the stream; the backend ID is only the call_id.
                    item_id = 'fc_' + uuid.uuid4().hex
                    self.calls[index] = {'id': call.get('id', ''), 'item_id': item_id,
                                         'name': function.get('name', ''),
                                         'arguments': function.get('arguments', ''),
                                         'output_index': len(self.response['output'])}
                    item = {'type': 'function_call', 'id': item_id, 'call_id': self.calls[index]['id'],
                            'name': self.calls[index]['name'], 'arguments': self.calls[index]['arguments'],
                            'status': 'in_progress'}
                    self.response['output'].append(item)
                    events.append(self.event('response.output_item.added',
                                             output_index=self.calls[index]['output_index'], item=item))
                    if item['arguments']:
                        events.append(self.event('response.function_call_arguments.delta',
                                                 output_index=self.calls[index]['output_index'],
                                                 item_id=item_id, delta=item['arguments']))
                assembled = self.calls[index]
                if call.get('id'):
                    assembled['id'] = call['id']
                    self.response['output'][assembled['output_index']]['call_id'] = call['id']
                function = call.get('function', {})
                name_delta = '' if is_new else function.get('name', '')
                argument_delta = '' if is_new else function.get('arguments', '')
                assembled['name'] += name_delta
                assembled['arguments'] += argument_delta
                item = self.response['output'][assembled['output_index']]
                item['name'] = assembled['name']
                item['arguments'] = assembled['arguments']
                if argument_delta:
                    events.append(self.event('response.function_call_arguments.delta',
                                             output_index=assembled['output_index'],
                                             item_id=assembled['item_id'], delta=argument_delta))
        return events

    def process_events(self, batch):
        events = []
        for event in batch.events:
            if isinstance(event, StartedEvent):
                self.response['model'] = event.model
            elif isinstance(event, TextDelta):
                events.extend(self.add_text(event.text))
            elif isinstance(event, ReasoningDelta):
                events.extend(self._add_reasoning(event.text))
            elif isinstance(event, ToolCallDelta):
                index = event.index
                if index not in self.calls:
                    self.calls[index] = {'id': event.call_id, 'item_id': 'fc_' + uuid.uuid4().hex,
                                         'name': event.name, 'arguments': '',
                                         'output_index': len(self.response['output'])}
                    item = {'type': 'function_call', 'id': self.calls[index]['item_id'],
                            'call_id': event.call_id, 'name': event.name, 'arguments': '', 'status': 'in_progress'}
                    self.response['output'].append(item)
                    events.append(self.event('response.output_item.added', output_index=self.calls[index]['output_index'], item=item))
                call = self.calls[index]
                call['arguments'] += event.arguments_delta
                item = self.response['output'][call['output_index']]
                item['arguments'] = call['arguments']
                if event.arguments_delta:
                    events.append(self.event('response.function_call_arguments.delta', output_index=call['output_index'], item_id=call['item_id'], delta=event.arguments_delta))
            elif isinstance(event, DoneEvent):
                self.finish_reason = event.finish_reason
            elif isinstance(event, UsageEvent):
                self.token_usage = event.usage
        return events

    def _add_reasoning(self, text):
        if self.reasoning is None:
            self.reasoning = {'type': 'reasoning', 'id': 'rs_' + uuid.uuid4().hex, 'summary': [],
                              'content': [{'type': 'reasoning_text', 'text': ''}]}
            self.reasoning_index = len(self.response['output'])
            self.response['output'].append(self.reasoning)
            events = [self.event('response.output_item.added', output_index=self.reasoning_index, item={**self.reasoning, 'content': []}),
                      self.event('response.content_part.added', output_index=self.reasoning_index, item_id=self.reasoning['id'], content_index=0, part=self.reasoning['content'][0])]
        else:
            events = []
        self.reasoning['content'][0]['text'] += text
        events.append(self.event('response.reasoning_summary_text.delta', output_index=self.reasoning_index, item_id=self.reasoning['id'], content_index=0, delta=text))
        return events

    def finish(self):
        if self.finish_reason is None:
            raise RuntimeError('Chat stream ended without a finish reason')
        converted = []
        if self.calls:
            # Calls are assembled first; backend may only identify them at the end.
            raw_calls = list(self.calls.values())
            converted_items = output_calls(self.request, [
                {k: call[k] for k in ('id', 'name', 'arguments')}
                for call in raw_calls
            ])
            converted = list(zip(raw_calls, converted_items))
            for call, item in converted:
                # Keep compatibility with callers that inject assembled calls
                # directly (for example adapters and unit tests).
                index = call.get('output_index', len(self.response['output']))
                if index == len(self.response['output']):
                    self.response['output'].append(item)
                self.response['output'][index] = item
                if call.get('item_id'):
                    # Keep the item ID stable across the stream; the backend
                    # ID only identifies the call (call_id).
                    item['id'] = call['item_id']
        # Item lifecycle events must arrive in output order; the terminal
        # chunks can finalize items in any order.
        groups = {}
        if self.calls:
            for call, item in converted:
                index = call.get('output_index', len(self.response['output']) - 1)
                if item['type'] == 'custom_tool_call':
                    group = [
                        self.event('response.custom_tool_call_input.delta', output_index=index, item_id=item['id'], delta=item['input']),
                        self.event('response.custom_tool_call_input.done', output_index=index, item_id=item['id'], input=item['input'])]
                else:
                    group = [self.event('response.function_call_arguments.done', output_index=index, item_id=item['id'],
                                        arguments=item['arguments'], name=item['name'])]
                group.append(self.event('response.output_item.done', output_index=index, item=item))
                groups.setdefault(index, []).extend(group)
        elif self.pending_text or not self.response['output']:
            # Buffered text creates its item now; its open events join the
            # item's group so the stream stays in output order.
            groups[self.message_index] = self.add_text(self.pending_text)
        if self.reasoning is not None:
            part = self.reasoning['content'][0]
            groups.setdefault(self.reasoning_index, []).extend([
                self.event('response.reasoning_summary_text.done', output_index=self.reasoning_index,
                           item_id=self.reasoning['id'], content_index=0, text=part['text']),
                self.event('response.content_part.done', output_index=self.reasoning_index,
                           item_id=self.reasoning['id'], content_index=0, part=part),
                self.event('response.output_item.done', output_index=self.reasoning_index, item=self.reasoning)])
        if self.message is not None:
            part = self.message['content'][0]
            self.message['status'] = 'incomplete' if self.finish_reason == 'length' else 'completed'
            groups.setdefault(self.message_index, []).extend([
                self.event('response.output_text.done', output_index=self.message_index,
                           item_id=self.message['id'], content_index=0, text=part['text'], logprobs=[]),
                self.event('response.content_part.done', output_index=self.message_index,
                           item_id=self.message['id'], content_index=0, part=part),
                self.event('response.output_item.done', output_index=self.message_index, item=self.message)])
        events = [event for index in sorted(groups) for event in groups[index]]
        complete_response(self.response, self.finish_reason, self.token_usage)
        save_response(self.request, self.response, self.history, self.store)
        events.append(self.event('response.' + self.response['status'], response=self.response))
        return events

    def failed(self, message, code='server_error'):
        self.response['status'] = 'failed'
        self.response['error'] = {'code': code, 'message': message}
        return self.event('response.failed', response=self.response)
