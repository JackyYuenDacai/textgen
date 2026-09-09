# Live full-context test — 2026-09-09

All three requests completed on the user's already loaded
Qwen3.8-27B-EXL3-3.5bpw model. The active capacity was 260,096 tokens, with
Q4 cache, four MTP draft tokens, 4,096-token prefill chunks and a 4 GiB
recurrent-checkpoint budget. No server restart, model reload or configuration
change was performed for this test.

| Request | Input tokens | Cached tokens | Prefill | Decode | Output tokens | HTTP elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold full context | 259,963 | 0 | 213.655 s | 74.47 tokens/s | 128 | 216.03 s |
| Same prompt repeated | 259,963 | 259,840 | 0.289 s | 91.14 tokens/s | 128 | 2.38 s |
| Exact input boundary | 260,090 | 259,840 | 0.546 s | 22.15 tokens/s | 1 | 1.21 s |

The first two prompts leave 128 output tokens and five internal headroom
tokens. At the exact boundary, the loader capped the requested 128 output
tokens to one. All requests reported zero dropped input tokens. The
one-token decode measurement is too short for a meaningful speed comparison.

Six allocator pressure reclaims occurred during cold prefill. The same prompt
subsequently reused 259,840 tokens (99.95%), demonstrating that these reclaims
did not purge its prefix cache. CUDA reported zero allocator retries and zero
OOMs in all completed-request snapshots.

Five-second NVIDIA samples reached 23,022 MiB used (22.48 GiB), with a minimum
of 1,014 MiB free. These are sampled values, not continuous peaks. At the end,
PyTorch reserved 21,134 MiB, live tensors occupied about 17,929 MiB, and CUDA
reported 1,513 MiB free. CUDA/PyTorch and NVIDIA tool accounting differ, so
their memory fields should not be treated as interchangeable.

The test used synthetic text through `/v1/completions` and verified generation,
prefix reuse, memory pressure handling and exact-boundary output budgeting.
It did not use real image encoding, test answer quality, or run a prolonged
soak test. No native crash occurred; this does not establish that the earlier
tokenizer access violation is eliminated.

Detailed request responses, performance snapshots, and GPU samples are in
`user_data/logs/full_context_live_20260909_205126/`. The reusable test is
`scripts/probe_full_context.py`; it sends live requests and populates the
server's prefix cache. Its prompts contain no user conversation data.

Separately, the staged CUDA startup checks and optional xFormers compatibility
guard passed all 65 ExLlamaV3 regression tests. These changes were made after
the current model loaded and take effect on a subsequent server restart.
