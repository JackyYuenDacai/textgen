# ExLlamaV3 performance controls

The native ExLlamav3 loader reuses unchanged images and reports generation timings separately from prompt processing. These changes apply to web chat and the local API.

## Image reuse

The image cache retains complete embedding objects, including their synthetic token IDs and position metadata. It separately keeps a small identity record for each image for the lifetime of the loaded model, even after that image's tensors are evicted. Re-encoding an unchanged image restores its original IDs, token list and alias, so eviction alone no longer changes the tokenized prompt prefix. Keys include decoded pixels, image dimensions/mode, the vision model instance and preprocessing settings. Changing the image or preprocessing creates a new identity; a changed token layout or position geometry also prevents reuse of incompatible IDs. Model unload clears both the tensor cache and identity records without resetting the global token allocator.

The default tensor limit is **256 MiB**, with at most 32 retained entries. Tensor backing allocations are counted, including deepstack tensors. The installed Qwen vision encoder returns these tensors in CPU memory. Each request identifies all its images before encoding starts: resident images needed anywhere in that request are protected, and eviction selects only other least-recently-used entries. If no unprotected entry can make room, the new embedding is used for that request without being retained. Duplicate images encode at most once per request. Eviction does not modify embeddings held by an active request.

The tensor limit is a retained-cache budget, not a limit on all embeddings needed by an active request. Identity records contain only hashes, integers and aliases (no images or tensors); their count grows with distinct images until model unload. Retaining these small records is what preserves token IDs across eviction. An over-budget history may still need some image re-encoding, but should retain its existing image hits and stable prompt IDs instead of repeatedly evicting the entire image history.

Set **Model → Image embedding cache (MiB)**, save model settings, and reload to change the limit. Set it to 0 to disable tensor retention across requests; token identities still remain stable. The CLI equivalent is `--exl3-image-cache-mib 256`. After updating the Python implementation, restart TextGen to load the code; changing the cache size alone does not reload Python modules. A restart initially starts with cold caches.

## Stable prompts and context controls

Tool definitions are rendered in a consistent order, including their JSON object keys. This covers both local/MCP tools and API tool definitions. Array order inside each schema, message order, system instructions, and saved chat contents are preserved. If a caller changes actual instructions or tool definitions, the affected prompt prefix must still be processed again.

Under **Parameters → Generation**, the **32k context**, **64k context**, and **Full context** buttons set the existing prompt truncation limit. They respect the loaded model's capacity. Shorter context omits old history from the request; it does not delete saved messages or generate a summary. Full context restores the loaded model's limit. The current Qwen configuration retains its 128,000-token default; use a shorter limit when old details are no longer needed.

## Reading the metrics

The console now labels its original overall speed as `tokens/s end-to-end`. A separate `ExLlamaV3 job` line reports backend decode speed, prefill seconds, queue seconds, image preparation seconds, cached/input tokens, and accepted/attempted draft tokens.

`GET /v1/internal/model/info` additionally returns `performance`:

- `recent_requests`: up to 32 request records, each with a job ID and completion status. Completed requests include backend timing, generated/emitted token counts, cache hits and draft acceptance.
- `image_cache`: retained entry count, tensor bytes, byte limit and `token_identities` count. Request-level `image_cache_hits` counts embedding reuse, not re-encodings that only recover the same token IDs.
- `max_chunk_size`: the active prompt processing chunk size.

Request metrics contain no prompt, image or generated text. Cancelled/interrupted requests do not inherit a previous request's completion timing. Backend generated counts may include a stop token; API usage also retokenizes visible text and can differ. The wrapper now consumes text, token IDs and logprobs in the final event before ending the stream, and counts tensor elements rather than the tensor's batch dimension.

## Settings selected on this machine

- Qwen3.8-27B-EXL3-3.5bpw, Q4 KV cache, 128,000 context.
- DFlash2 EXL3 5.0bpw, seven draft tokens. A fixed draft size required by the model is applied automatically before its cache is loaded.
- Image cache: 256 MiB.
- Prefill chunk size: **2,048**. Two fresh approximately 42k-token requests averaged 16.05s prefill at 2,048 and 15.96s at 4,096. That difference is too small to justify increasing the default. The Model tab exposes the setting for later workloads; larger chunks also inform model loading so temporary memory is accounted for.
- Unrecognized native ExLlamaV3 cache formats fail with an explanatory error before model allocation. `fp8` and `nvfp4` do not silently fall back to FP16. Supported formats are `fp16`, `q2` through `q8`, and mixed integer formats such as `q4_q8`.

## Validation on 2026-09-05

Twelve regression tests passed:

```powershell
.\installer_files\env\python.exe -X utf8 -m unittest discover -s tests -p test_exllamav3_performance.py -v
```

Live API checks verified image reuse and invalidation, model reload clearing, successful text/image output, metrics, and identical chat prompts when tool lists are shuffled. A synthetic image request used 3,776 prompt tokens: its first-output delay decreased from **1.35s to 0.19s** when repeated, with **3,584 cached tokens**. Image preparation dropped from **0.032s to 0.003s** and prefill from **1.28s to 0.128s**. Decode stayed about 95–96 tokens/s; the gain was avoiding repeated image/prompt work. Changing one pixel caused a miss and reprocessing as expected. These are targeted tests, not a promise for every conversation.

Raw results and reproducible local-API probes are under `downloads/verify_performance_changes.*` and `downloads/benchmark_prefill_chunks.*`. These probes reload the model and populate its prompt cache. The chunk benchmark restores 2,048 afterward. The running server's output is in `downloads/performance-server.log`; it uses `user_data/cache/triton` for compiled kernels.
