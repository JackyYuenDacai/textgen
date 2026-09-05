"""Validate optional drafting controls without importing the GPU backend."""

import inspect
import math


def drafting_options(args, generator_class):
    enabled = args.exl3_dynamic_draft
    if not isinstance(enabled, bool):
        raise ValueError('exl3-dynamic-draft must be a boolean.')
    confidence = float(args.exl3_draft_confidence)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('exl3-draft-confidence must be strictly between 0 and 1.')
    if not enabled:
        return {}
    parameters = inspect.signature(generator_class).parameters
    if not {'dynamic_draft_tokens', 'draft_confidence'} <= parameters.keys():
        raise ValueError('This ExLlamaV3 version does not support adaptive drafting. Upgrade ExLlamaV3 or disable adaptive draft length.')
    return {'dynamic_draft_tokens': True, 'draft_confidence': confidence}
