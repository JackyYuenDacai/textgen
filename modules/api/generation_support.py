import copy
import functools
import json
import time
from collections import deque
from pathlib import Path

import tiktoken
import yaml
from pydantic import ValidationError

from .errors import InvalidRequestError
from .typing import ToolDefinition
from .utils import debug_msg
from modules.tool_parsing import get_tool_call_id, parse_tool_call, detect_tool_call_format
from modules import shared, utils
from modules.reasoning import extract_reasoning
from modules.chat import (
    generate_chat_prompt,
    generate_chat_reply,
    load_character_memoized,
    load_instruction_template_memoized
)
from modules.image_utils import convert_openai_messages_to_images
from modules.logging_colors import logger
from modules.presets import load_preset_memoized
from modules.text_generation import decode, encode, generate_reply


@functools.cache
def load_chat_template_file(filepath):
    """Load a chat template from a file path (.jinja, .jinja2, or .yaml/.yml)."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding='utf-8')
    if filepath.suffix.lower() in utils.YAML_EXTENSIONS:
        data = yaml.safe_load(text) or {}
        return data.get('instruction_template', '')
    return text


def _first_token_display_str(token_id, prompt, tokenizer):
    """Return the display string for the first prompt token.

    Returns empty string for BOS or tokens that don't appear at the start
    of the prompt text, so they don't shift text_offset for subsequent tokens.
    """
    token_id = int(token_id)
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    if bos_id is not None and token_id == bos_id:
        return ""

    import torch
    tok = tokenizer.decode(torch.tensor([token_id]))
    if not prompt.startswith(tok):
        return ""

    return tok


def _compute_prompt_logprob_entries(prompt, logprobs_count, input_ids=None):
    """Compute logprob entries for prompt tokens via a forward pass.

    Returns a list of logprob entries in the standard format.
    The first token gets a null entry (no conditioning context).

    Supported for HF-compatible loaders (Transformers, ExLlamav3_HF, etc.)
    via a single forward pass, and for llama.cpp via the server's
    prompt_logprobs parameter. Returns [] for unsupported loaders.
    """
    if input_ids is None:
        input_ids = encode(prompt)  # (1, seq_len) tensor or array

    token_ids = input_ids[0]
    n_tokens = len(token_ids)

    if n_tokens == 0:
        return []

    loader = shared.args.loader
    model = shared.model

    if loader == 'llama.cpp':
        return model.get_prompt_logprob_entries(token_ids, max(logprobs_count, 1), prompt=prompt)

    first_token_str = _first_token_display_str(token_ids[0], prompt, shared.tokenizer)

    if n_tokens <= 1:
        return [{"token": first_token_str, "null_logprob": True}]

    import torch
    from modules.torch_utils import clear_torch_cache

    if hasattr(model, 'get_prompt_logits'):
        logits = model.get_prompt_logits(input_ids)

    elif hasattr(model, 'forward'):
        # HF-compatible loaders (Transformers, etc.). Loaders that need a
        # custom path (e.g. wrappers that only compute last-token logits in
        # __call__) should expose get_prompt_logits() above.
        input_ids_tensor = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
        if hasattr(model, 'device'):
            input_ids_tensor = input_ids_tensor.to(model.device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids_tensor)
            logits = outputs.logits  # keep on device, (1, seq_len, vocab) in model dtype
            del outputs

    else:
        return []

    entries = [{"token": first_token_str, "null_logprob": True}]

    logprobs_count = max(logprobs_count, 1)
    k = min(logprobs_count, logits.shape[-1])
    chunk_size = 2048
    unique_ids = set(int(tid) for tid in token_ids[1:])

    # Process logits in chunks, only move top-K results to CPU
    all_top_log_probs_list = []
    all_top_indices_list = []
    all_actual_lps = []

    for start in range(0, n_tokens - 1, chunk_size):
        end = min(start + chunk_size, n_tokens - 1)
        chunk_logits = logits[0, start:end].float()  # (chunk, vocab) on logits.device
        chunk_lse = torch.logsumexp(chunk_logits, dim=-1)
        chunk_top_values, chunk_top_indices = torch.topk(chunk_logits, k=k, dim=-1)
        chunk_top_log_probs = chunk_top_values - chunk_lse.unsqueeze(-1)

        # Compute logprob for actual next tokens in this chunk
        chunk_top_sets = [set(chunk_top_indices[j].tolist()) for j in range(end - start)]
        for j in range(end - start):
            actual_tid = int(token_ids[start + j + 1])
            if actual_tid not in chunk_top_sets[j]:
                all_actual_lps.append((chunk_logits[j, actual_tid] - chunk_lse[j]).item())
            else:
                all_actual_lps.append(None)  # will use top_log_probs

        all_top_log_probs_list.append(chunk_top_log_probs.cpu())
        all_top_indices_list.append(chunk_top_indices.cpu())
        unique_ids.update(int(tid) for tid in chunk_top_indices.flatten().tolist())
        del chunk_logits, chunk_lse, chunk_top_values

    del logits
    clear_torch_cache()

    all_top_log_probs = torch.cat(all_top_log_probs_list, dim=0)
    all_top_indices = torch.cat(all_top_indices_list, dim=0)

    unique_ids_list = sorted(unique_ids)
    decoded_list = shared.tokenizer.batch_decode([[tid] for tid in unique_ids_list]) if hasattr(shared.tokenizer, 'batch_decode') else [shared.tokenizer.decode(torch.tensor([tid])) for tid in unique_ids_list]
    decoded_strs = dict(zip(unique_ids_list, decoded_list))

    for i in range(1, n_tokens):
        token_id = int(token_ids[i])
        idx = i - 1
        top_log_probs = all_top_log_probs[idx]
        top_ids = all_top_indices[idx].tolist()
        actual_token_str = decoded_strs[token_id]

        if token_id in top_ids:
            actual_lp = top_log_probs[top_ids.index(token_id)].item()
            alternatives = [
                {"token": decoded_strs[top_ids[j]], "token_id": top_ids[j], "logprob": top_log_probs[j].item()}
                for j in range(k) if top_ids[j] != token_id
            ]
        else:
            actual_lp = all_actual_lps[idx]
            alternatives = [
                {"token": decoded_strs[top_ids[j]], "token_id": top_ids[j], "logprob": top_log_probs[j].item()}
                for j in range(k - 1)
            ]

        entry = {"top_logprobs": [{"token": actual_token_str, "token_id": token_id, "logprob": actual_lp}] + alternatives}
        entries.append(entry)

    return entries


def _get_raw_logprob_entries(offset=0):
    """Get raw logprob entries from llama.cpp/ExLlamav3 backend, starting from offset.

    Returns (new_entries, new_offset).
    """
    if not hasattr(shared.model, 'last_completion_probabilities') or not shared.model.last_completion_probabilities:
        return [], offset

    all_entries = shared.model.last_completion_probabilities
    new_entries = all_entries[offset:]
    return new_entries, len(all_entries)


def _dict_to_logprob_entries(token_dict):
    """Convert a flat {token: logprob} dict (from LogprobProcessor) to raw entry format."""
    if not token_dict:
        return []

    return [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in token_dict.items()]}]


def _parse_entry_top(entry):
    """Extract the top logprobs list from a raw entry, handling both key names."""
    return entry.get('top_logprobs', entry.get('top_probs', []))


def _extract_sampled_token(entry, top):
    """Get the actually sampled token and its logprob from a logprob entry.

    Uses the entry-level token/logprob when available (the actually sampled
    token), falling back to top[0] (highest-probability alternative) which
    may differ with non-greedy sampling.
    """
    if 'token' in entry:
        return entry['token'], entry.get('logprob', entry.get('prob', 0))

    token_str = top[0].get('token', '')
    token_logprob = top[0].get('logprob', top[0].get('prob', 0))
    return token_str, token_logprob


def format_chat_logprobs(entries):
    """Format logprob entries into OpenAI chat completions logprobs format.

    Output: {"content": [{"token", "logprob", "bytes", "top_logprobs": [...]}]}
    """
    if not entries:
        return None

    content = []
    for entry in entries:
        top = _parse_entry_top(entry)
        if not top:
            continue

        token_str, token_logprob = _extract_sampled_token(entry, top)

        top_list = []
        for item in top:
            t = item.get('token', '')
            lp = item.get('logprob', item.get('prob', 0))
            top_list.append({
                "token": t,
                "logprob": lp,
                "bytes": list(t.encode('utf-8')) if t else None
            })

        content.append({
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_str.encode('utf-8')) if token_str else None,
            "top_logprobs": top_list
        })

    return {"content": content, "refusal": None} if content else None


def format_completion_logprobs(entries):
    """Format logprob entries into OpenAI completions logprobs format.

    Output: {"tokens", "token_logprobs", "top_logprobs": [{token: prob}], "top_logprobs_ids": [{token_id: prob}], "text_offset"}
    """
    if not entries:
        return None

    tokens = []
    token_logprobs = []
    top_logprobs = []
    top_logprobs_ids = []
    text_offset = []
    offset = 0

    for entry in entries:
        # Handle null logprob entries (first prompt token with echo)
        if entry.get("null_logprob"):
            token_str = entry.get("token", "")
            tokens.append(token_str)
            token_logprobs.append(None)
            top_logprobs.append(None)
            top_logprobs_ids.append(None)
            text_offset.append(offset)
            offset += len(token_str)
            continue

        top = _parse_entry_top(entry)
        if not top:
            continue

        token_str, token_logprob = _extract_sampled_token(entry, top)

        tokens.append(token_str)
        token_logprobs.append(token_logprob)
        text_offset.append(offset)
        offset += len(token_str)

        top_dict = {}
        top_dict_ids = {}
        for item in top:
            t = item.get('token', '')
            lp = item.get('logprob', item.get('prob', 0))
            top_dict[t] = lp
            tid = item.get('token_id', item.get('id'))
            if tid is not None:
                top_dict_ids[tid] = lp
        top_logprobs.append(top_dict)
        top_logprobs_ids.append(top_dict_ids if top_dict_ids else None)

    if not tokens:
        return None

    result = {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset
    }
    if any(x is not None for x in top_logprobs_ids):
        result["top_logprobs_ids"] = top_logprobs_ids
    return result


def process_parameters(body, is_legacy=False):
    generate_params = body
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    generate_params['max_new_tokens'] = body.pop(max_tokens_str)
    if generate_params['truncation_length'] == 0:
        generate_params['truncation_length'] = shared.settings['truncation_length']

    if generate_params['temperature'] == 0:
        generate_params['do_sample'] = False
        generate_params['top_k'] = 1

    if body['preset'] is not None:
        preset = load_preset_memoized(body['preset'])
        generate_params.update(preset)

    generate_params['custom_stopping_strings'] = []
    if 'stop' in body:  # str or array, max len 4 (ignored)
        if isinstance(body['stop'], str):
            generate_params['custom_stopping_strings'] = [body['stop']]
        elif isinstance(body['stop'], list):
            generate_params['custom_stopping_strings'] = body['stop']

    # Resolve logprobs: for chat completions, logprobs is a bool and the count
    # comes from top_logprobs. Normalize to an int for all backends.
    logprobs = body.get('logprobs', None)
    top_logprobs = body.get('top_logprobs', None)
    if logprobs is True:
        logprobs = max(top_logprobs, 1) if top_logprobs is not None else 5
        generate_params['logprobs'] = logprobs

    # For llama.cpp and ExLlamav3 native, logit_bias and logprobs are forwarded natively
    if (shared.args.loader not in ('llama.cpp', 'ExLlamav3')
            and (body.get('logit_bias') or logprobs is not None)):
        from transformers import LogitsProcessorList

        from modules.transformers_loader import (
            LogitsBiasProcessor,
            LogprobProcessor
        )

        logits_processor = []
        logit_bias = body.get('logit_bias', None)
        if logit_bias:  # {str: float, ...}
            logits_processor = [LogitsBiasProcessor(logit_bias)]

        if logprobs is not None and logprobs > 0:
            generate_params['logprob_proc'] = LogprobProcessor(logprobs)
            logits_processor.extend([generate_params['logprob_proc']])

        if logits_processor:  # requires logits_processor support
            generate_params['logits_processor'] = LogitsProcessorList(logits_processor)

    return generate_params


def process_multimodal_content(content):
    """Extract text and add image placeholders from OpenAI multimodal format"""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        image_placeholders = ""
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get('type', '')
            if item_type == 'text':
                text_parts.append(item.get('text', ''))
            elif item_type == 'image_url':
                image_placeholders += "<__media__>"

        final_text = '\n'.join(text_parts)
        if image_placeholders:
            return f"{image_placeholders}\n\n{final_text}"
        else:
            return final_text

    return str(content)


def convert_history(history, preserve_response_items=False):
    '''
    Chat histories in this program are in the format [message, reply].
    This function converts OpenAI histories to that format.
    '''
    chat_dialogue = []
    current_message = ""
    current_reply = ""
    user_input = ""
    user_input_last = True
    system_message = ""
    seen_non_system = False

    for entry in history:
        content = process_multimodal_content(entry.get("content"))
        role = entry["role"]

        if role == "user":
            seen_non_system = True
            user_input = content
            user_input_last = True

            if current_message:
                chat_dialogue.append([current_message, '', '', {}])
                current_message = ""

            current_message = content
        elif role == "assistant":
            seen_non_system = True
            meta = {}
            tool_calls = entry.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                meta["tool_calls"] = tool_calls
                if content.strip() == "":
                    content = ""  # keep empty content, don't skip

            if preserve_response_items:
                meta['response_assistant'] = {'role': 'assistant', 'content': content}
                for key in ('reasoning_content', 'tool_calls', 'phase'):
                    if key in entry:
                        meta['response_assistant'][key] = copy.deepcopy(entry[key])

            current_reply = content
            user_input_last = False
            if current_message:
                chat_dialogue.append([current_message, current_reply, '', meta])
                current_message = ""
                current_reply = ""
            else:
                chat_dialogue.append(['', current_reply, '', meta])
        elif role == "tool":
            seen_non_system = True
            user_input_last = False
            meta = {}
            if "tool_call_id" in entry:
                meta["tool_call_id"] = entry["tool_call_id"]
            if preserve_response_items:
                meta['response_tool'] = True  # Empty function output is still a tool result.
            chat_dialogue.append(['', '', content, meta])
        elif role in ("system", "developer"):
            if not seen_non_system:
                # Leading system messages go to custom_system_message (placed at top)
                system_message += f"\n{content}" if system_message else content
            else:
                # Mid-conversation system messages: preserve position in history
                if current_message:
                    chat_dialogue.append([current_message, '', '', {}])
                    current_message = ""
                chat_dialogue.append([content, '', '', {"role": "system"}])

    if not user_input_last:
        user_input = ""

    return user_input, system_message, {
        'internal': chat_dialogue,
        'visible': copy.deepcopy(chat_dialogue),
        'messages': history  # Store original messages for multimodal models
    }


def validateTools(tools: list[dict]):
    # Validate each tool definition in the JSON array
    valid_tools = None
    for idx in range(len(tools)):
        tool = tools[idx]
        try:
            tool_definition = ToolDefinition(**tool)
            # Backfill defaults so Jinja2 templates don't crash on missing fields
            func = tool.get("function", {})
            if "description" not in func:
                func["description"] = ""
            if "parameters" not in func:
                func["parameters"] = {"type": "object", "properties": {}}
            if valid_tools is None:
                valid_tools = []
            valid_tools.append(tool)
        except ValidationError:
            raise InvalidRequestError(message=f"Invalid tool specification at index {idx}.", param='tools')

    return valid_tools
