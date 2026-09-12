"""Lossless conversation items; only the prompt renderer flattens model turns."""
import copy
import json
from dataclasses import dataclass
from .errors import InvalidRequestError


def invalid(message, param='input', code=400):
    raise InvalidRequestError(message=message, param=param, code=code)


@dataclass
class GenerationItem:
    data: dict

    def to_dict(self):
        return copy.deepcopy(self.data)


class MessageItem(GenerationItem): pass
class ReasoningItem(GenerationItem): pass
class FunctionCallItem(GenerationItem): pass
class FunctionCallOutputItem(GenerationItem): pass
class CustomToolCallItem(GenerationItem): pass
class CustomToolCallOutputItem(GenerationItem): pass


ITEM_TYPES = dict(message=MessageItem, reasoning=ReasoningItem,
                  function_call=FunctionCallItem, function_call_output=FunctionCallOutputItem,
                  custom_tool_call=CustomToolCallItem, custom_tool_call_output=CustomToolCallOutputItem)


@dataclass
class ConversationState:
    items: list[GenerationItem]

    @classmethod
    def from_items(cls, items):
        return cls([ITEM_TYPES[item.get('type', 'message')](copy.deepcopy(item)) for item in items])

    def to_dicts(self):
        return [item.to_dict() for item in self.items]


def from_chat_messages(messages):
    items = []
    for message in messages:
        role = message['role']
        if role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                          'output': message['content']})
            continue
        if message.get('reasoning_content'):
            items.append({'type': 'reasoning', 'content': [
                {'type': 'reasoning_text', 'text': message['reasoning_content']}]})
        content = message.get('content')
        if content is not None:
            if isinstance(content, list):
                content = [({'type': 'input_image', **part['image_url']} if part['type'] == 'image_url'
                            else {'type': 'input_text', 'text': part['text']}) for part in content]
                for part in content:
                    if part['type'] == 'input_image':
                        part['image_url'] = part.pop('url')
            item = {'type': 'message', 'role': role, 'content': content}
            if message.get('phase'):
                item['phase'] = message['phase']
            items.append(item)
        for call in message.get('tool_calls', []):
            items.append({'type': 'function_call', 'call_id': call['id'], **call['function']})
    return ConversationState.from_items(items).items


def _string(item, key, param, allow_empty=False):
    value = item.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        invalid(f'{key} must be a string' + ('.' if allow_empty else ' and cannot be empty.'), param)
    return value


def _fields(item, allowed, param):
    unknown = set(item) - set(allowed.split())
    if unknown:
        invalid('Unsupported fields: ' + ', '.join(sorted(unknown)), param)


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


def render_items(items):
    items = [item.to_dict() if isinstance(item, GenerationItem) else item for item in items] if not isinstance(items, str) else items
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


