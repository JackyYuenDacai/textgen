# Draft tuning results — 2026-09-05

Applied: **fixed `draft-max=4`**, with adaptive drafting available but disabled.

Hardware: RTX 5090 D v2, 24 GB. Model: Qwen3.8-27B-EXL3-3.5bpw.
Backend: ExLlamaV3 1.4.6. Context allocation stayed at 260,096 tokens; KV cache
stayed Q4. No model precision, thinking defaults, or GPU power settings were changed.

## Results

Decode throughput, weighted by generated tokens / decode time:

| Configuration | First sweep (tok/s) | Confirmation (tok/s) |
| --- | ---: | ---: |
| Fixed 2 | 121.93 | — |
| Fixed 3 (original) | 135.48 | 135.42 |
| Fixed 4 (selected) | 141.94 | 140.34 |
| Fixed 5 | 99.16 | — |
| Adaptive 3, target 0.40 | 135.75 | — |
| Adaptive 4, target 0.40 | — | 139.76 |
| Adaptive 5, target 0.40 | 118.54 | — |

Fixed 4 improved overall decode throughput by **3.6–4.8%** in the two comparisons.
In the first sweep, short coding improved from 144.76 to 156.75 tok/s and maths
from 147.76 to 156.38 tok/s. The synthetic long-context case changed from 118.10
to 119.60 tok/s: essentially no material gain. Adaptive 4 was close to fixed 4,
with no demonstrated advantage. Higher acceptance alone was not a useful ranking:
fixed 2 had higher acceptance but lower throughput. Length 5 was slower in both modes.

## Protocol and limitations

- API Chat Completions through one TextGen server. Only one request at a time.
- Three identical prompts per configuration: coding (50 prompt tokens), maths (67),
  and synthetic worker-pool context (19,318).
- Temperature 0, seed 42, thinking disabled **for these requests only**; output cap
  512 tokens with EOS allowed. Three unmeasured 128-token warmups, then two measured
  rounds per profile.
- Each profile reloaded the model, retaining the same context/cache allocation.
  Prompt-cache hits were allowed after warmup; decode and request timings are
  recorded separately. Adaptive calibration continued across warmup and measured samples.
- The confirmation order was adaptive 4 → fixed 4 → fixed 3, providing a later
  repeat of both the selected fixed profile and the original baseline.
- Generated text was not bit-identical across all repeats/configurations despite
  greedy settings. These are throughput measurements on matched prompts and output
  budgets, not identical-token replay or a correctness/capability comparison.
- This small test does not establish performance on the user's 100K-token chats,
  sampled/thinking-enabled output, or arbitrary coding tasks. Close differences
  should be treated as noise until tested on more representative requests.
- An earlier separate-process attempt produced an abnormal 4–5 tok/s baseline
  while the old server also retained a GPU context. It was aborted and excluded.
  Valid comparisons were run through the restarted server itself.

## Local artifacts

Raw settings, per-answer hashes/text, acceptance, timings, and summaries are under:

- `user_data/benchmarks/draft-sweep-20260905-2252/`
- `user_data/benchmarks/draft-confirm-20260905-2258/`

The original saved loader settings are retained in each directory's `restore.json`.
The original live performance snapshot was retained before reloading. Reloads reset
in-memory prompt/image caches and live performance history, not saved chat files.

See [Draft tuning](Draft-tuning.md) for the controls and repeatable benchmark command.
