# ExLlamaV3 draft tuning

Under **Model → Speculative decoding**, ExLlamaV3 supports:

- **draft-max**: the fixed draft length, or the maximum length when adaptive drafting is enabled.
- **Adaptive draft length (ExLlamaV3)**: use the backend's online confidence calibration to stop drafting early when acceptance is unlikely.
- **Adaptive draft acceptance target**: strictly between 0 and 1; default 0.40. Higher values prune more aggressively. This is not a guaranteed measured acceptance rate.

Click **Load** to apply changes and **Save settings** to persist them. A server restart
is required once after upgrading TextGen to make newly added controls available.
Equivalent flags are `--draft-max`, `--exl3-dynamic-draft`, and
`--exl3-draft-confidence`. The last two settings apply only to the native ExLlamaV3
loader. Defaults preserve fixed drafting. Older backends can still use fixed drafting;
requesting unsupported adaptive drafting fails with an explicit upgrade message.

A loaded external drafter or enabled built-in MTP head is required. The built-in
MTP head explains why acceptance can appear without a separately selected draft model.
The Performance tab identifies the drafting source, fixed/adaptive mode, and maximum
length; the internal model-info API exposes the same settings in `performance.drafting`.

The maximum draft length also determines recurrent rollback-state allocation.
Changing it requires reloading the model, even when adaptive drafting is enabled.
Longer drafts can use more VRAM and can be slower despite high acceptance.
Adaptive drafting changes the proposed verification window, not the target sampler;
it does not intentionally lower model precision or replace target-model verification.
Floating-point differences between verification shapes can still change generated text.

## Reproducible local comparison

Use TextGen's Python interpreter, while no chat, API client, or capability benchmark
is generating. This script reloads the local API model, compares fixed
lengths 2–5 and adaptive lengths 3–5 at target 0.40 through that same server, then reloads
the original model using its **saved** settings. Save your current Model settings
beforehand if you have unsaved changes. The API does not expose every live loading
parameter, so the script cannot reconstruct unsaved settings. LoRA runs are refused.

```powershell
.\installer_files\env\python.exe -X utf8 -m scripts.benchmark_drafting --model YOUR_MODEL --output user_data/benchmarks/draft-sweep-UNIQUE --allow-model-reload
```

`--api-base` defaults to `http://127.0.0.1:5000/v1`. For an authenticated API set
`TEXTGEN_BENCH_API_KEY` and, if different, `TEXTGEN_BENCH_ADMIN_KEY` in the process
environment; neither is stored in reports.
Use `--profiles adaptive-4 fixed-4 fixed-3` to run a smaller confirmation sweep.
The script does not change saved model settings. Browser chat history is not deleted,
but model reloads clear prompt/image caches and reset live performance history.
If the controller is forcibly terminated, restore the model via the Model tab or
the model-load API using the saved `restore.json`.

The protocol uses identical coding, maths, and synthetic long-context prompts,
greedy generation, thinking disabled, a 512-token output cap with EOS allowed,
three 128-token warmups, and two measured rounds. Context allocation, cache format,
and model precision stay fixed. Reports include actual prompt lengths, generated text,
answer hashes, acceptance, per-request timings, and settings.
Aggregate decode speed is total output tokens divided by total decode time, not an
unweighted mean of request speeds. This is a small throughput comparison, not a
capability evaluation or proof of speed on 100K-token conversations. Repeat close
results and compare representative workloads before choosing a default.
