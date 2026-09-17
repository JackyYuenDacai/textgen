"""Bounded, pre-delivery recovery for client-executed tool turns.

No HTTP recursion, tool execution, or persistent synthetic conversation state.
Tool-enabled turns are staged so a rejected attempt can never be executed.
"""
import copy
import json
import logging
import os
import re

from .canonical_items import from_chat_messages
from .generation_events import DoneEvent, ToolCallDelta, TextDelta, UsageEvent
from .responses_tools import ToolOutputError, strict_validator
from modules.reasoning import extract_reasoning
from modules.tool_parsing import TOOL_CALL_OPENING_MARKERS


logger = logging.getLogger(__name__)

def malformed_output(answer, calls, names, parsers, parse):
    """Check attempted calls before the generation layer hides their markup."""
    _, content = extract_reasoning(answer)
    # Examples in Markdown are not evidence of an attempted action.
    visible = re.sub(r'```.*?(?:```|$)|`[^`\n]*`', '', content, flags=re.S)
    if '<tool_call>' in visible:
        blocks = visible.split('<tool_call>')[1:]
        if any('</tool_call>' not in block or
               not parse('<tool_call>' + block, names, parsers=parsers)
               for block in blocks):
            return 'An attempted tool-call batch contains an invalid or unfinished call.'
    if not calls and (any(marker in visible for marker in TOOL_CALL_OPENING_MARKERS)
                      or any(re.search(r'\b' + re.escape(name) + r'\s*\{', visible) for name in names)):
        return 'Attempted tool output could not be parsed as a declared tool call.'
    return None


def progress_only(text):
    """Conservative early-EOS heuristic, not a general task-completion judge."""
    text = text.strip()
    if not text or len(text) > 600 or any(c in text for c in ('?', '？', '`', '\n')):
        return False
    if re.search(r"\b(cannot|can't|unable|blocked|need your|please provide)\b|无法|不能|需要你", text, re.I):
        return False
    # Require a short first-person action promise, not a factual final answer.
    return bool(re.fullmatch(
        r"(?:(?:Okay|OK|Sure)[,.!]?\s*)?(?:I(?:'ll| will| am going to)|Let me) "
        r"(?:(?:now|next|continue to) )?(?:check|inspect|search|fetch|process|implement|"
        r"fix|run|test|create|update|continue|complete|work on)\b[^.!?]*(?:[.!])?",
        text, re.I) or re.fullmatch(
        r'(?:好的[，,。]?\s*)?(?:我(?:会|将|现在|接下来)|接下来我(?:会|将)?)'
        r'(?:继续|检查|查询|搜索|处理|实现|修复|运行|测试|创建|更新|完成)[^。！？]*(?:。)?', text))


def _validate_calls(body, calls):
    if not body.get('_responses_parallel_tool_calls', body.get('parallel_tool_calls', True)) and len(calls) > 1:
        raise ToolOutputError('Generate at most one tool call in this response.')
    if '_responses_tool_validation' in body:
        from types import SimpleNamespace
        from .responses import output_calls
        output_calls(SimpleNamespace(**body['_responses_tool_validation']), calls)
        return
    validator = body.get('_validate_generation_calls')
    if validator:
        validator(calls)
        return
    # Chat Completions supplies nested function definitions.
    definitions = {t['function']['name']: t['function'] for t in body.get('tools', []) if 'function' in t}
    for call in calls:
        definition = definitions.get(call['name'])
        if definition is None:
            raise ToolOutputError('Generated an undeclared tool.')
        if definition.get('strict'):
            try:
                strict_validator(definition).validate(json.loads(call['arguments']))
            except Exception as exc:
                raise ToolOutputError('Generated arguments do not match the strict schema.') from exc


def recover(source, body, *, is_legacy=False, stream=False, stop_event=None):
    """Consume attempts privately; expose only an accepted turn and total usage."""
    max_repairs = max(0, min(2, int(os.environ.get('TEXTGEN_TOOL_REPAIR_ATTEMPTS', '2'))))
    detect_progress = os.environ.get('TEXTGEN_REPAIR_PROGRESS', '1').lower() not in ('0', 'false', 'off')
    original_items = body['items']
    attempt_body = dict(body)
    total = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    cached_tokens = 0
    fingerprints = set()
    budget = body.get('length' if is_legacy else 'max_tokens')
    for attempt in range(max_repairs + 1):
        if stop_event is not None and stop_event.is_set():
            return
        metrics = attempt_body.get('_responses_metrics')
        if metrics is not None:
            metrics.clear()
        detail = {}
        attempt_body['_generation_attempt'] = detail
        if budget is not None:
            attempt_body['_generation_output_ceiling'] = budget - total['completion_tokens']
        generator = source(attempt_body, is_legacy=is_legacy, stream=stream, stop_event=stop_event)
        try:
            batches = list(generator)
        finally:
            generator.close()
        if stop_event is not None and stop_event.is_set():
            return
        if detail.get('cancelled'):
            return
        events = [event for batch in batches for event in batch.events]
        usage = next((e.usage for e in events if isinstance(e, UsageEvent)), None)
        if usage is None:
            raise ToolOutputError('Generation ended without usage; refusing an unbounded retry.')
        for key in total:
            total[key] += usage.get(key, 0)
        cached_tokens += usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if budget is None:
            budget = detail.get('output_budget', usage['completion_tokens'])
        calls = [dict(id=e.call_id, name=e.name, arguments=e.arguments_delta)
                 for e in events if isinstance(e, ToolCallDelta)]
        finish = next((e for e in events if isinstance(e, DoneEvent)), None)
        if finish is None:
            raise ToolOutputError('Generation ended without a terminal event.')
        error = detail.get('error')
        if not error and calls:
            try:
                _validate_calls(body, calls)
            except ToolOutputError as exc:
                error = str(exc)
        text = ''.join(e.text for e in events if isinstance(e, TextDelta))
        if not error and not calls and detect_progress and progress_only(text):
            error = 'The response only promises future work without performing it.'
        if not error and not calls and not text.strip():
            error = 'The response contains no visible answer or executable tool call.'
        exhausted = finish.finish_reason == 'length' or (error is not None and total['completion_tokens'] >= budget)
        if error and not exhausted:
            fingerprint = (error, detail.get('raw', text).strip())
            if attempt >= max_repairs or fingerprint in fingerprints:
                raise ToolOutputError(f'Generation repair exhausted after {attempt} repair attempt(s): {error}')
            fingerprints.add(fingerprint)
            logger.info('Server generation repair attempt=%d remaining_output_tokens=%d reason=%s',
                        attempt + 1, budget - total['completion_tokens'], error)
            # Only the current rejected attempt is retained. Neither it nor
            # this correction enters the public response store.
            correction = (
                'The previous response was not delivered and no tools from it were executed. '
                + error + ' Correct the response now using the original task and actual tool results. '
                'If more work remains, issue valid calls to the available tools in the native tool format. '
                'Regenerate the entire intended call batch; do not repeat a promise to act. '
                'Do not invent tool results or missing arguments. If genuinely blocked, explain the blocker '
                'or ask the necessary question. If the task is already done, give the concrete final result.')
            attempt_body['items'] = list(original_items) + from_chat_messages([
                {'role': 'assistant', 'content': detail.get('raw', text)},
                {'role': 'user', 'content': correction}])
            continue
        # A truncated batch is never executable, even if its first call parsed.
        if exhausted:
            finish.finish_reason = 'length'
            for batch in batches:
                batch.events = [e for e in batch.events if not isinstance(e, ToolCallDelta)]
        final_usage = dict(total)
        if cached_tokens:
            final_usage['prompt_tokens_details'] = {'cached_tokens': cached_tokens}
        for batch in batches:
            for event in batch.events:
                if isinstance(event, UsageEvent):
                    event.usage = copy.deepcopy(final_usage)
            yield batch
        return
