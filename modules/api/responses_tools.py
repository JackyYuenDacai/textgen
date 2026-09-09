"""Local validation for Responses tool schemas and free-form tool input.

Validation happens before exposing a completed executable tool call to clients.
It does not provide constrained decoding or execute the tool.
"""
import json
from functools import lru_cache


class ToolOutputError(ValueError):
    """The backend generated an unusable tool call, rather than invalid input."""


@lru_cache(maxsize=32)
def grammar_parser(syntax, definition):
    if not isinstance(definition, str) or not definition or len(definition) > 65536:
        raise ValueError('Tool grammar must contain 1 to 65536 characters.')
    if syntax == 'regex':
        import regex
        return regex.compile(definition)
    if syntax == 'lark':
        # Imported grammars must not read arbitrary server-side files.
        import re
        from lark import Lark
        for source in re.findall(r'%import\s+([^\s(]+)', definition):
            if source != 'common' and not source.startswith('common.'):
                raise ValueError('Only Lark common grammar imports are supported.')
        return Lark(definition, parser='earley')
    raise ValueError('Tool grammar syntax must be lark or regex.')


def check_custom_format(tool):
    fmt = tool.get('format') or {'type': 'text'}
    if not isinstance(fmt, dict):
        raise ValueError('Custom tool format must be an object.')
    if fmt == {'type': 'text'}:
        return
    if fmt.get('type') != 'grammar' or set(fmt) != {'type', 'syntax', 'definition'}:
        raise ValueError('Custom tool format must be text or a grammar definition.')
    grammar_parser(fmt['syntax'], fmt['definition'])


def custom_function(tool):
    check_custom_format(tool)
    description = (tool.get('description') or '') + '\nSupply the complete raw tool input in the input string, preserving newlines.'
    fmt = tool.get('format') or {}
    if fmt.get('type') == 'grammar':
        description += '\nThe raw input must match this ' + fmt['syntax'] + ' grammar:\n' + fmt['definition']
    return {'type': 'function', 'name': tool['name'], 'description': description, 'strict': False,
            'parameters': {'type': 'object', 'properties': {'input': {'type': 'string'}},
                           'required': ['input'], 'additionalProperties': False}}


def custom_input(tool, arguments):
    try:
        data = json.loads(arguments)
    except (ValueError, TypeError) as exc:
        raise ValueError('Custom tool call did not generate valid JSON transport arguments.') from exc
    if not isinstance(data, dict) or set(data) != {'input'} or not isinstance(data['input'], str):
        raise ValueError('Custom tool call must contain one input string.')
    text = data['input']
    fmt = tool.get('format') or {}
    if fmt.get('type') == 'grammar':
        if len(text) > 262144:
            raise ValueError('Custom tool input exceeds the local grammar validation limit.')
        parser = grammar_parser(fmt['syntax'], fmt['definition'])
        if fmt['syntax'] == 'regex':
            try:
                match = parser.fullmatch(text, timeout=0.25)
            except TimeoutError as exc:
                raise ValueError('Custom tool regex validation exceeded its time limit.') from exc
            if match is None:
                raise ValueError('Custom tool input does not match the requested regex grammar.')
        else:
            try:
                parser.parse(text)
            except Exception as exc:
                raise ValueError('Custom tool input does not match the requested Lark grammar.') from exc
    return text


@lru_cache(maxsize=32)
def schema_validator(serialized):
    from jsonschema.validators import validator_for
    schema = json.loads(serialized)
    def local_refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('$ref', '$dynamicRef') and (not isinstance(item, str) or not item.startswith('#')):
                    raise ValueError('Tool schemas support local references only.')
                local_refs(item)
        elif isinstance(value, list):
            for item in value: local_refs(item)
    local_refs(schema)
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def strict_validator(tool):
    return schema_validator(json.dumps(tool.get('parameters') or {'type': 'object', 'properties': {}}, sort_keys=True))
