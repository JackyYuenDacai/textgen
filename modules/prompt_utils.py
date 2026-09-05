"""Keep equivalent tool definitions stable when constructing prompt prefixes."""
import json


def stable_tool_definitions(tools):
    if not isinstance(tools, list) or not tools:
        return tools

    # JSON object key order is not semantic. Preserve all array order within a
    # schema (e.g. examples, enum, prefixItems), and never mutate caller data.
    serialized = [json.dumps(tool, sort_keys=True, ensure_ascii=False) for tool in tools]
    return [json.loads(tool) for tool in sorted(serialized)]
