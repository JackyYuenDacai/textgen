"""Responses structured output: schema compilation and terminal validation."""
import json
import re
from functools import lru_cache

from .errors import InvalidRequestError, OpenAIError
from .responses_tools import schema_validator


class StructuredOutputError(OpenAIError):
    def __init__(self):
        super().__init__('Generated output did not match text.format.schema.', code=500)


@lru_cache(maxsize=32)
def compile_schema(serialized):
    from llguidance import LLMatcher
    schema_validator(serialized)
    # Do not silently ignore keywords unsupported by the grammar compiler.
    grammar = LLMatcher.grammar_from_json_schema(serialized)
    error = LLMatcher.validate_grammar(grammar)
    if error:
        raise ValueError(error)
    return grammar


def prepare_format(text, loader):
    fmt = (text or {}).get('format', {'type': 'text'})
    if fmt == {'type': 'text'}:
        return None
    def invalid(message):
        raise InvalidRequestError(message, 'text.format')
    if not isinstance(fmt, dict) or fmt.get('type') != 'json_schema':
        invalid('Supported text formats are text and json_schema.')
    if set(fmt) - {'type', 'name', 'description', 'strict', 'schema'}:
        invalid('Unsupported json_schema format fields.')
    if not isinstance(fmt.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', fmt['name']):
        invalid('json_schema requires a name of 1-64 letters, digits, underscores or hyphens.')
    if fmt.get('strict') is not None and type(fmt['strict']) is not bool:
        invalid('json_schema strict must be boolean or null.')
    if 'description' in fmt and not isinstance(fmt['description'], str):
        invalid('json_schema description must be a string.')
    if not isinstance(fmt.get('schema'), dict) or fmt['schema'].get('type') != 'object':
        invalid('json_schema requires an object schema at the root.')
    if loader != 'ExLlamav3':
        invalid('json_schema constrained decoding currently requires the ExLlamav3 loader.')
    try:
        return compile_schema(json.dumps(fmt['schema'], sort_keys=True))
    except ImportError:
        invalid('json_schema requires llguidance; install requirements/responses.txt in the server environment.')
    except Exception:
        invalid('Invalid or unsupported JSON Schema for local constrained decoding.')


def validate_output(request, response):
    fmt = (request.text or {}).get('format') or {}
    if fmt.get('type') != 'json_schema' or response['status'] == 'incomplete':
        return
    try:
        if any(item['type'] != 'message' for item in response['output']):
            raise ValueError('Unexpected non-message output')
        output = ''.join(part['text'] for item in response['output']
                         for part in item['content'] if part['type'] == 'output_text')
        def reject_constant(value):
            raise ValueError('Non-JSON numeric constant')
        value = json.loads(output, parse_constant=reject_constant)
        schema_validator(json.dumps(fmt['schema'], sort_keys=True)).validate(value)
    except Exception:
        raise StructuredOutputError() from None
