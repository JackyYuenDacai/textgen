"""Keep reusable prompt prefixes stable."""
import json
import re


_CURRENT_TIME = re.compile(r'<current_time>[^<>]*</current_time>')


def relocate_current_time(messages):
    """Move leading system time metadata behind history without changing its value.

    Only exact, complete tags with plain-text contents are metadata here. Never
    rewrite user/assistant/tool content or mutate the caller's messages. Keep
    trailing assistant messages last so assistant prefills still work.
    """
    result = list(messages)
    time_messages = []
    leading_count = 0
    for index, message in enumerate(messages):
        if message.get('role') not in ('system', 'developer'):
            break
        leading_count += 1
        content = message.get('content')
        if not isinstance(content, str):
            continue
        blocks = _CURRENT_TIME.findall(content)
        if blocks:
            # Preserve surrounding whitespace and the original system message,
            # even if empty, so the prefix stays stable between requests.
            result[index] = dict(message, content=_CURRENT_TIME.sub('', content))
            time_messages.append({'role': message['role'], 'content': '\n'.join(blocks)})

    if not time_messages or leading_count == len(messages):
        return messages

    insert_at = len(result)
    while insert_at > leading_count and result[insert_at - 1].get('role') == 'assistant':
        insert_at -= 1
    result[insert_at:insert_at] = time_messages
    return result


def stable_tool_definitions(tools):
    if not isinstance(tools, list) or not tools:
        return tools

    # JSON object key order is not semantic. Preserve all array order within a
    # schema (e.g. examples, enum, prefixItems), and never mutate caller data.
    serialized = [json.dumps(tool, sort_keys=True, ensure_ascii=False) for tool in tools]
    return [json.loads(tool) for tool in sorted(serialized)]
