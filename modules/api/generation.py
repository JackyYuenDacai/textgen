"""Protocol-independent local generation and semantic output events.

Tool recognition remains in the model parser. A tool-capable parser may need
the complete call before it can distinguish markup from ordinary text.
"""
from enum import Enum, auto
from .generation_events import (EventBatch, StartedEvent, TextDelta, ReasoningDelta,
                                ToolCallDelta, DoneEvent, UsageEvent)
from .canonical_items import render_items
from .generation_support import *
from .generation_support import (_get_raw_logprob_entries, _dict_to_logprob_entries,
                                _compute_prompt_logprob_entries)


class GenerationState(Enum):
    GENERATING_REASONING = auto()
    GENERATING_TEXT = auto()
    GENERATING_TOOL_CALL = auto()
    WAITING_FOR_TOOL_RESULT = auto()
    COMPLETED = auto()

def stream(body: dict, is_legacy: bool = False, stream=False, prompt_only=False, stop_event=None) -> dict:
    if body.get('functions', []):
        raise InvalidRequestError(message="functions is not supported.", param='functions')

    if body.get('function_call', ''):
        raise InvalidRequestError(message="function_call is not supported.", param='function_call')

    body = dict(body)
    messages = render_items(body.pop('items'))
    tools = None
    if 'tools' in body and body['tools'] is not None and isinstance(body['tools'], list) and body['tools']:
        tools = validateTools(body['tools'])  # raises InvalidRequestError if validation fails

    tool_choice = body.get('tool_choice', None)
    if tool_choice == "none":
        tools = None  # Disable tool detection entirely

    for m in messages:
        if 'role' not in m:
            raise InvalidRequestError(message="messages: missing role", param='messages')
        elif m['role'] == 'function':
            raise InvalidRequestError(message="role: function is not supported.", param='messages')

        # Handle multimodal content validation
        content = m.get('content')
        if content is None:
            # OpenAI allows content: null on assistant messages when tool_calls is present
            if m['role'] == 'assistant' and m.get('tool_calls'):
                m['content'] = ''
            else:
                raise InvalidRequestError(message="messages: missing content", param='messages')

        # Validate multimodal content structure
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or 'type' not in item:
                    raise InvalidRequestError(message="messages: invalid content item format", param='messages')
                if item['type'] not in ['text', 'image_url']:
                    raise InvalidRequestError(message="messages: unsupported content type", param='messages')
                if item['type'] == 'text' and 'text' not in item:
                    raise InvalidRequestError(message="messages: missing text in content item", param='messages')
                if item['type'] == 'image_url' and ('image_url' not in item or 'url' not in item['image_url']):
                    raise InvalidRequestError(message="messages: missing image_url in content item", param='messages')

    # generation parameters
    generate_params = process_parameters(body, is_legacy=is_legacy)
    if stop_event is not None:
        generate_params['stop_event'] = stop_event
    continue_ = body['continue_']

    # Instruction template
    if body['instruction_template_str']:
        instruction_template_str = body['instruction_template_str']
    elif body['instruction_template']:
        instruction_template = body['instruction_template']
        instruction_template = "Alpaca" if instruction_template == "None" else instruction_template
        instruction_template_str = load_instruction_template_memoized(instruction_template)
    elif shared.args.chat_template_file:
        instruction_template_str = load_chat_template_file(shared.args.chat_template_file)
    else:
        instruction_template_str = shared.settings['instruction_template_str']

    chat_template_str = body['chat_template_str'] or shared.default_settings['chat_template_str']
    chat_instruct_command = body['chat_instruct_command'] or shared.default_settings['chat-instruct_command']

    # Chat character
    character = body['character'] or shared.default_settings['character']
    character = "Assistant" if character == "None" else character
    name1 = body['user_name'] or shared.default_settings['name1']
    name1, name2, _, greeting, context = load_character_memoized(character, name1, '')
    name2 = body['bot_name'] or name2
    context = body['context'] or context
    greeting = body['greeting'] or greeting
    user_bio = body['user_bio'] or ''

    # History
    user_input, custom_system_message, history = convert_history(messages, body.get('_responses_preserve_items', False))

    generate_params.update({
        'mode': body['mode'],
        'name1': name1,
        'name2': name2,
        'context': context,
        'greeting': greeting,
        'user_bio': user_bio,
        'instruction_template_str': instruction_template_str,
        'custom_system_message': custom_system_message,
        'cache_friendly_current_time': body.get('cache_friendly_current_time', True),
        'chat_template_str': chat_template_str,
        'chat-instruct_command': chat_instruct_command,
        'tools': tools,
        'history': history,
        'stream': stream
    })

    max_tokens = generate_params['max_new_tokens']
    if max_tokens is not None and max_tokens <= 0:
        raise InvalidRequestError(message="max_tokens must be greater than 0.", param="max_tokens")

    if max_tokens is None:
        generate_params['max_new_tokens'] = 512
        generate_params['auto_max_new_tokens'] = True

    requested_model = generate_params.pop('model')
    logprob_proc = generate_params.pop('logprob_proc', None)
    if logprob_proc:
        logprob_proc.token_alternatives_history.clear()
    chat_logprobs_offset = [0]  # mutable for closure access in streaming

    def semantic_batch(content=None, chunk_tool_calls=None, include_role=False, reasoning_content=None):
        events = []
        if include_role:
            events.append(StartedEvent(shared.model_name))
        if reasoning_content is not None:
            events.append(ReasoningDelta(reasoning_content))
        if content is not None:
            events.append(TextDelta(content))
        for index, call in enumerate(chunk_tool_calls or []):
            events.append(ToolCallDelta(call['id'], call['function']['name'],
                                        call['function']['arguments'], call.get('index', index)))
        batch = EventBatch(events)
        if logprob_proc:
            entries = _dict_to_logprob_entries(logprob_proc.token_alternatives)
            formatted = format_chat_logprobs(entries)
            if formatted:
                batch.logprobs = formatted
        elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
            entries, chat_logprobs_offset[0] = _get_raw_logprob_entries(chat_logprobs_offset[0])
            if entries:
                formatted = format_chat_logprobs(entries)
                if formatted:
                    batch.logprobs = formatted

        return batch

    # generate reply #######################################
    if prompt_only:
        prompt = generate_chat_prompt(user_input, generate_params, _continue=continue_)
        yield {'prompt': prompt}
        return

    if stream:
        chunk = semantic_batch('', include_role=True)
        yield chunk

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=False, _continue=continue_, loading_message=False)

    answer = ''
    state = GenerationState.GENERATING_TEXT
    seen_content = ''
    seen_reasoning = ''

    tool_calls = []
    end_last_tool_call = 0
    supported_tools = [x["function"]["name"] for x in tools] if tools is not None else None
    _tool_parsers = None

    # Filter supported_tools when tool_choice specifies a particular function
    if supported_tools and isinstance(tool_choice, dict):
        specified_func = tool_choice.get("function", {}).get("name")
        if specified_func and specified_func in supported_tools:
            supported_tools = [specified_func]

    if supported_tools is not None:
        _template_str = generate_params.get('instruction_template_str', '') if generate_params.get('mode') == 'instruct' else generate_params.get('chat_template_str', '')
        _tool_parsers, _, _ = detect_tool_call_format(_template_str)

    try:
        for a in generator:
            answer = a['internal'][-1][1]

            if supported_tools is not None:
                tool_call = parse_tool_call(answer[end_last_tool_call:], supported_tools, parsers=_tool_parsers) if len(answer) > 0 else []
                if len(tool_call) > 0:
                    for tc in tool_call:
                        tc["id"] = get_tool_call_id()
                        if stream:
                            tc["index"] = len(tool_calls)
                        tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
                        tool_calls.append(tc)
                    state = GenerationState.GENERATING_TOOL_CALL
                    end_last_tool_call = len(answer)

            # Stop generation before streaming content if tool_calls were detected,
            # so that raw tool markup is not sent as content deltas.
            if len(tool_calls) > 0:
                break

            if stream:
                # Strip reasoning/thinking blocks so only final content is streamed.
                # Reasoning is emitted separately as reasoning_content deltas.
                reasoning, content = extract_reasoning(answer)
                if reasoning is not None:
                    state = GenerationState.GENERATING_REASONING
                    new_reasoning = reasoning[len(seen_reasoning):]
                    new_content = content[len(seen_content):]
                else:
                    state = GenerationState.GENERATING_TEXT
                    new_reasoning = None
                    new_content = answer[len(seen_content):]

                if (not new_content and not new_reasoning) or chr(0xfffd) in (new_content or '') + (new_reasoning or ''):
                    continue

                chunk = semantic_batch(
                    content=new_content if new_content else None,
                    reasoning_content=new_reasoning if new_reasoning else None,
                )

                if reasoning is not None:
                    seen_reasoning = reasoning
                    seen_content = content
                else:
                    seen_content = answer
                yield chunk
    finally:
        # Responses can stop as soon as a complete tool call is recognized.
        # Release the backend job before yielding the terminal API events,
        # including when the consumer closes or generation raises.
        if generate_params.get('_responses_raise_errors'):
            generator.close()

    response_metrics = generate_params.get('_responses_metrics', {})
    if 'prompt_tokens' in response_metrics:
        token_count = response_metrics['prompt_tokens']
        completion_token_count = response_metrics['completion_tokens']
    else:
        token_count = shared.model.last_prompt_token_count if hasattr(shared.model, 'last_prompt_token_count') else 0
        completion_token_count = len(encode(answer)[0])
    if len(tool_calls) > 0:
        stop_reason = "tool_calls"
        state = GenerationState.WAITING_FOR_TOOL_RESULT
    elif response_metrics.get('finish_reason'):
        stop_reason = response_metrics['finish_reason']
    elif token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= generate_params['max_new_tokens']:
        stop_reason = "length"
    else:
        stop_reason = "stop"
        state = GenerationState.COMPLETED

    token_usage = {
        'prompt_tokens': token_count,
        'completion_tokens': completion_token_count,
        'total_tokens': token_count + completion_token_count,
    }
    if 'cached_tokens' in response_metrics:
        token_usage['prompt_tokens_details'] = {'cached_tokens': response_metrics['cached_tokens']}

    if stream:
        batch = semantic_batch(chunk_tool_calls=tool_calls)
        batch.events.append(DoneEvent(stop_reason))
        yield batch
        yield EventBatch([UsageEvent(token_usage)])
    else:
        reasoning, content = extract_reasoning(answer)
        batch = semantic_batch(content=None if tool_calls else content,
                               reasoning_content=reasoning if reasoning else None,
                               chunk_tool_calls=tool_calls, include_role=True)
        if logprob_proc:
            all_entries = []
            for alt in logprob_proc.token_alternatives_history:
                all_entries.extend(_dict_to_logprob_entries(alt))
            batch.logprobs = format_chat_logprobs(all_entries)
        elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
            raw = getattr(shared.model, 'last_completion_probabilities', None)
            if raw:
                batch.logprobs = format_chat_logprobs(raw)
        batch.events.extend([DoneEvent(stop_reason), UsageEvent(token_usage)])
        yield batch


def collect(body, stop_event=None):
    return list(stream(body, stream=False, stop_event=stop_event))
