# Prefill memory and cache tuning

These changes take effect on the next server restart. No restart, model reload, synthetic inference, or cache flush is performed by the tuning tools. Keep the user's selected context capacity fixed throughout comparisons.

## Changes

- Quantized prefill staging uses 64-page / 16384-token allocation increments above small prompt sizes. At 131433 tokens, a single sequence with four KV heads of dimension 256 needs 576 MiB of combined FP16 K/V staging instead of 1024 MiB. This only changes unused allocation padding; it does not trim context or change attention inputs. The allocation is transient, but PyTorch may retain its memory. More size classes may increase fragmentation; this tradeoff must be measured.
- `EXL3_QC_STAGING_BUCKET_PAGES=0` restores the old power-of-two allocation policy for comparison. The default is 64. This environment setting is read on backend import, so changing it requires restarting the process.
- Model controls expose the system-RAM checkpoint budget, tail/decode checkpoint interval, and prefill checkpoint interval. The saved Qwen configuration uses an 8192-token prefill interval instead of the backend default 32768, retaining the 4096 MiB budget and 2048 tail interval. The actual prefill interval rounds up to a chunk boundary. More checkpoints can reduce replay when branching, but can also cause more host copies and evictions. They cannot reuse an actually changed prompt prefix.
- Per-request metrics include configuration, uncached prompt throughput, memory before generation, first text output, consumer exit, and checkpoint-cache occupancy/eviction counters for completed jobs. These are available through the existing performance API. The Performance tab shows chunk size, uncached prefill rate, and allocated/reserved/free memory at exit.

## Interpreting memory

All byte fields are literal bytes. CUDA counters are process-wide, and device free memory includes other applications. Peaks explicitly named `process_peak_*` are cumulative process peaks, not request peaks. Samples do not synchronize CUDA or reset counters. A cancelled request's exit sample can precede native cleanup. Concurrent requests can overlap any sample, so do not infer per-job allocation ownership from these measurements. A missing or failed measurement is not zero usage.

Recurrent-cache eviction counters are cumulative within the generator. Compare their deltas between completed requests. Checkpoints are in system RAM. Memory reserved minus allocated includes reusable/fragmented allocator memory; it is not proof of a leak. Increasing live allocations under comparable repeated workloads is a reason to investigate further.

## Comparison after the user restarts

1. Keep context, Q4, draft settings, and model unchanged. Start with the saved 4096-token chunk size and 8192-token prefill checkpoint interval. Let ordinary requests populate metrics.
2. Export snapshots with the existing read-only performance probe. Save separate files per session. No automatic benchmarking or reloading is provided.
3. At a user-chosen reload, compare 2048 versus 4096 chunks; compare 8192 versus 32768 prefill checkpoint intervals separately. Changing one variable at a time avoids confounding allocation, checkpoint, and chunk effects. A model reload makes the next prompt cold.
4. Run `installer_files\env\python.exe scripts\summarize_prefill.py snapshot-2048.json snapshot-4096.json --out comparison.csv`.

The offline tool deduplicates overlapping snapshots with session/sequence IDs and groups by model, context, cache type, chunk, checkpoint intervals, checkpoint RAM budget, staging policy, drafting settings, cold/warm cache, 16k prompt-length band and 4k uncached-length band. Compare equivalent groups; rates are total uncached tokens / total prefill seconds, not an average of ratios. Prompt content and machine load still differ, so these observations are not a controlled benchmark. Minimum sampled free memory can miss transient peaks.

Keep 4096 only if equivalent workloads show a benefit without increasing allocator retries or memory pressure. Keep denser checkpoints only if reduced replay outweighs copying and eviction. The winner cannot be established until observations are available after restart. Do not use per-token `empty_cache()` as a performance fix.

The modified backend file and new staging helper are staged in both the local ExLlamaV3 source and this workspace's installed Python environment. Upgrading/reinstalling ExLlamaV3 may overwrite the installed copies; retain the source changes.
