"""Offline CPU tokenizer stress check; no server calls or model weights loaded.

Run in the server's Python environment. Uses synthetic text only. Native
crashes terminate this subprocess, so run separately from the server.
"""
import argparse
import faulthandler
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

# Must precede torch/backend imports. This probe must not allocate GPU memory.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from exllamav3.tokenizer import Tokenizer
from exllamav3.tokenizer.mm_embedding import MMEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--tokens', type=int, default=220000)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--synthetic-images', type=int, default=40,
                        help='Use 0 to encode a single unsplit long text prompt.')
    args = parser.parse_args()
    if args.tokens < 1 or args.rounds < 1:
        parser.error('--tokens and --rounds must be positive')
    if args.synthetic_images < 0:
        parser.error('--synthetic-images must be nonnegative')
    faulthandler.enable(all_threads=True)
    tokenizer = Tokenizer(SimpleNamespace(
        directory=args.model_dir, bos_token_id=None, eos_token_id=None,
        pad_token_id=None, eos_token_id_list=[],
    ))
    phrase = 'Tokenizer stress: English 中文 日本語 🙂 code(x) = 123;\n'
    unit = tokenizer.encode(phrase).numel()
    text = phrase * (args.tokens // max(1, unit) + 1)
    special = next(iter(tokenizer.extended_piece_to_id), '')
    # Synthetic embeddings exercise alias splitting and billion-range IDs;
    # they do not run the vision encoder or allocate any VRAM.
    embeddings = [MMEmbedding(embeddings=torch.zeros(2, 4),
                              token_string=torch.tensor([[-1, -1]])) for _ in range(args.synthetic_images)]
    stride = max(1, len(text) // max(1, len(embeddings)))
    chunks = [text[i:i + stride] for i in range(0, len(text), stride)]
    prompt = ''.join(chunk + (embeddings[i].text_alias if i < len(embeddings) else '')
                     for i, chunk in enumerate(chunks)) + special
    expected = {mode: tokenizer.encode(prompt, encode_special_tokens=mode, embeddings=embeddings)
                for mode in (False, True)}
    started = time.perf_counter()

    def run(worker):
        mode = bool(worker % 2)
        for _ in range(args.rounds):
            actual = tokenizer.encode(prompt, encode_special_tokens=mode, embeddings=embeddings)
            if not torch.equal(actual, expected[mode]):
                raise AssertionError('Concurrent token IDs differ from serial result')
            tokenizer.decode(actual[0], decode_special_tokens=True)
            tokenizer.num_tokens(phrase)
        return args.rounds

    with ThreadPoolExecutor(max_workers=4) as pool:
        completed = sum(pool.map(run, range(4)))
    print(json.dumps({
        'result': 'passed', 'concurrent_round_trips': completed,
        'tokens_per_prompt': {str(k): v.numel() for k, v in expected.items()},
        'synthetic_images': len(embeddings), 'seconds': round(time.perf_counter() - started, 3),
        'cuda_initialized': torch.cuda.is_initialized(),
    }), flush=True)


if __name__ == '__main__':
    main()
