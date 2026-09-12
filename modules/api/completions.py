"""Chat and legacy text-completion protocol adapters."""
from . import generation
from .canonical_items import from_chat_messages
from .generation_events import ChatSerializer
from .generation_support import *
from .generation_support import (_get_raw_logprob_entries, _dict_to_logprob_entries,
                                 _compute_prompt_logprob_entries)

def completions_common(body: dict, is_legacy: bool = False, stream=False, stop_event=None):
    object_type = 'text_completion'
    created_time = int(time.time())
    cmpl_id = "cmpl-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    prompt_str = 'context' if is_legacy else 'prompt'

    # Handle both prompt and messages format for unified multimodal support
    if prompt_str not in body or body[prompt_str] is None:
        if 'messages' in body:
            # Convert messages format to prompt for completions endpoint
            prompt_text = ""
            for message in body.get('messages', []):
                if isinstance(message, dict) and 'content' in message:
                    # Extract text content from multimodal messages
                    content = message['content']
                    if isinstance(content, str):
                        prompt_text += content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                prompt_text += item.get('text', '')

            # Allow empty prompts for image-only requests
            body[prompt_str] = prompt_text
        else:
            raise InvalidRequestError("Missing required input", param=prompt_str)

    # common params
    generate_params = process_parameters(body, is_legacy=is_legacy)
    max_tokens = generate_params['max_new_tokens']
    if max_tokens is None:
        generate_params['max_new_tokens'] = 512
        generate_params['auto_max_new_tokens'] = True
        max_tokens = 512
    elif max_tokens < 0:
        raise InvalidRequestError(message="max_tokens must be greater than or equal to 0.", param="max_tokens")
    elif max_tokens == 0 and body.get('logprobs') is None:
        raise InvalidRequestError(message="max_tokens is 0 but no logprobs parameter was specified.", param="max_tokens")

    generate_params['stream'] = stream
    if stop_event is not None:
        generate_params['stop_event'] = stop_event
    requested_model = generate_params.pop('model')
    logprob_proc = generate_params.pop('logprob_proc', None)
    if logprob_proc:
        logprob_proc.token_alternatives_history.clear()
    suffix = body['suffix'] if body['suffix'] else ''
    echo = body['echo']

    # Add messages to generate_params if present for multimodal processing
    if body.get('messages'):
        generate_params['messages'] = body['messages']
        raw_images = convert_openai_messages_to_images(generate_params['messages'])
        if raw_images:
            logger.info(f"Found {len(raw_images)} image(s) in request.")
            generate_params['raw_images'] = raw_images

    n_completions = body.get('n', 1) or 1

    if not stream:
        prompt_arg = body[prompt_str]

        # Handle empty/None prompts (e.g., image-only requests)
        if prompt_arg is None:
            prompt_arg = ""

        if isinstance(prompt_arg, str) or (isinstance(prompt_arg, list) and len(prompt_arg) > 0 and isinstance(prompt_arg[0], int)):
            prompt_arg = [prompt_arg]

        resp_list_data = []
        total_completion_token_count = 0
        total_prompt_token_count = 0
        choice_index = 0

        for idx, prompt in enumerate(prompt_arg, start=0):
            if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], int):
                # token lists
                if requested_model == shared.model_name:
                    prompt = decode(prompt)[0]
                else:
                    try:
                        encoder = tiktoken.encoding_for_model(requested_model)
                        prompt = encoder.decode(prompt)
                    except KeyError:
                        prompt = decode(prompt)[0]

            prefix = prompt if echo else ''
            prompt_input_ids = encode(prompt)
            token_count = len(prompt_input_ids[0])
            total_prompt_token_count += token_count

            # Compute prompt logprobs once per prompt (shared across n_completions)
            logprobs_val = body.get('logprobs', None)
            if echo and logprobs_val is not None and logprobs_val >= 0:
                prompt_entries = _compute_prompt_logprob_entries(prompt, logprobs_val, input_ids=prompt_input_ids)
            else:
                prompt_entries = None

            original_seed = generate_params.get('seed', -1)
            for _n in range(n_completions):
                # Increment seed for each completion to ensure diversity (matches llama.cpp native behavior)
                if original_seed >= 0:
                    generate_params['seed'] = original_seed + _n

                if logprob_proc:
                    logprob_proc.token_alternatives_history.clear()

                # generate reply #######################################
                if max_tokens == 0:
                    answer = ''
                    completion_token_count = 0
                    stop_reason = "stop"
                else:
                    debug_msg({'prompt': prompt, 'generate_params': generate_params})
                    generator = generate_reply(prompt, generate_params, is_chat=False)
                    answer = ''

                    for a in generator:
                        answer = a

                    completion_token_count = len(encode(answer)[0])
                    stop_reason = "stop"
                    if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
                        stop_reason = "length"

                total_completion_token_count += completion_token_count

                if max_tokens == 0:
                    all_entries = []
                else:
                    if logprob_proc:
                        all_entries = []
                        for alt in logprob_proc.token_alternatives_history:
                            all_entries.extend(_dict_to_logprob_entries(alt))
                    elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
                        all_entries = getattr(shared.model, 'last_completion_probabilities', None) or []
                    else:
                        all_entries = []

                if prompt_entries:
                    all_entries = prompt_entries + all_entries

                completion_logprobs = format_completion_logprobs(all_entries) if all_entries else None

                respi = {
                    "index": choice_index,
                    "finish_reason": stop_reason,
                    "text": prefix + answer + suffix,
                    "logprobs": completion_logprobs,
                }

                resp_list_data.append(respi)
                choice_index += 1

        resp = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            "system_fingerprint": None,
            resp_list: resp_list_data,
            "usage": {
                "prompt_tokens": total_prompt_token_count,
                "completion_tokens": total_completion_token_count,
                "total_tokens": total_prompt_token_count + total_completion_token_count
            }
        }

        yield resp
    else:
        prompt = body[prompt_str]
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                try:
                    encoder = tiktoken.encoding_for_model(requested_model)
                    prompt = encoder.decode(prompt)
                except KeyError:
                    prompt = decode(prompt)[0]
            else:
                raise InvalidRequestError(message="API Batched generation not yet supported.", param=prompt_str)

        prefix = prompt if echo else ''
        prompt_input_ids = encode(prompt)
        token_count = len(prompt_input_ids[0])

        # Check if usage should be included in streaming chunks per OpenAI spec
        stream_options = body.get('stream_options')
        include_usage = bool(stream_options) and bool(stream_options.get('include_usage') if isinstance(stream_options, dict) else getattr(stream_options, 'include_usage', False))
        cmpl_logprobs_offset = [0]  # mutable for closure access in streaming

        def text_streaming_chunk(content):
            # begin streaming
            if logprob_proc:
                chunk_logprobs = format_completion_logprobs(_dict_to_logprob_entries(logprob_proc.token_alternatives))
            elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
                entries, cmpl_logprobs_offset[0] = _get_raw_logprob_entries(cmpl_logprobs_offset[0])
                chunk_logprobs = format_completion_logprobs(entries) if entries else None
            else:
                chunk_logprobs = None

            chunk = {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": shared.model_name,
                "system_fingerprint": None,
                resp_list: [{
                    "index": 0,
                    "finish_reason": None,
                    "text": content,
                    "logprobs": chunk_logprobs,
                }],
            }

            return chunk

        logprobs_val = body.get('logprobs', None)
        if echo and logprobs_val is not None and logprobs_val >= 0:
            prompt_entries = _compute_prompt_logprob_entries(prompt, logprobs_val, input_ids=prompt_input_ids)
            prompt_logprobs_formatted = format_completion_logprobs(prompt_entries) if prompt_entries else None
        else:
            prompt_logprobs_formatted = None

        # Clear stale logprobs from any previous request before building the
        # first chunk, so text_streaming_chunk doesn't pick up old data.
        if hasattr(shared.model, 'last_completion_probabilities'):
            shared.model.last_completion_probabilities = []
        cmpl_logprobs_offset[0] = 0

        chunk = text_streaming_chunk(prefix)
        if prompt_logprobs_formatted is not None:
            chunk[resp_list][0]["logprobs"] = prompt_logprobs_formatted
        if include_usage:
            chunk['usage'] = None
        yield chunk

        # generate reply #######################################
        if max_tokens == 0:
            answer = ''
            completion_token_count = 0
            stop_reason = "stop"
        else:
            debug_msg({'prompt': prompt, 'generate_params': generate_params})
            generator = generate_reply(prompt, generate_params, is_chat=False)
            answer = ''
            seen_content = ''
            completion_token_count = 0

            for a in generator:
                answer = a

                len_seen = len(seen_content)
                new_content = answer[len_seen:]

                if not new_content or chr(0xfffd) in new_content:  # partial unicode character, don't send it yet.
                    continue

                seen_content = answer
                chunk = text_streaming_chunk(new_content)
                if include_usage:
                    chunk['usage'] = None
                yield chunk

            completion_token_count = len(encode(answer)[0])
            stop_reason = "stop"
            if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
                stop_reason = "length"

        chunk = text_streaming_chunk(suffix)
        chunk[resp_list][0]["finish_reason"] = stop_reason
        usage = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }

        if include_usage:
            chunk['usage'] = None
            yield chunk
            # Separate usage-only chunk with choices: [] per OpenAI spec
            yield {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": shared.model_name,
                "system_fingerprint": None,
                resp_list: [],
                "usage": usage
            }
        else:
            yield chunk


def chat_completions_common(body, is_legacy=False, stream=False, prompt_only=False, stop_event=None):
    body = dict(body)
    # Preserve the established monkey-patching/extension seam: generation
    # resolves backend hooks through this adapter's namespace.
    generation.generate_chat_reply = generate_chat_reply
    generation.parse_tool_call = parse_tool_call
    generation.get_tool_call_id = get_tool_call_id
    if 'items' not in body:
        if 'messages' not in body:
            raise InvalidRequestError(message='messages is required', param='messages')
        body['items'] = from_chat_messages(body.pop('messages'))
    serializer = ChatSerializer(stream=stream, is_legacy=is_legacy,
                                include_usage=bool(body.get('stream_options') and
                                    (body['stream_options'].get('include_usage') if isinstance(body['stream_options'], dict)
                                     else body['stream_options'].include_usage)))
    source = generation.stream(body, is_legacy=is_legacy, stream=stream,
                               prompt_only=prompt_only, stop_event=stop_event)
    try:
        for batch in source:
            if prompt_only:
                yield batch
            else:
                yield from serializer.process(batch)
    finally:
        source.close()


def chat_completions(body: dict, is_legacy: bool = False, stop_event=None) -> dict:
    generator = chat_completions_common(body, is_legacy, stream=False, stop_event=stop_event)
    return deque(generator, maxlen=1).pop()


def stream_chat_completions(body: dict, is_legacy: bool = False, stop_event=None):
    for resp in chat_completions_common(body, is_legacy, stream=True, stop_event=stop_event):
        yield resp


def completions(body: dict, is_legacy: bool = False, stop_event=None) -> dict:
    generator = completions_common(body, is_legacy, stream=False, stop_event=stop_event)
    return deque(generator, maxlen=1).pop()


def stream_completions(body: dict, is_legacy: bool = False, stop_event=None):
    for resp in completions_common(body, is_legacy, stream=True, stop_event=stop_event):
        yield resp


