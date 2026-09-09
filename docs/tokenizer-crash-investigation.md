# Tokenizer access violation investigation — 2026-09-09

The reported Windows exception `0xC0000005` is a native invalid-memory access.
The faulting instruction and leading frames are in `tokenizers.pyd`, running
under Python 3.13.15 with tokenizers 0.23.2. The exported `PyInit_*` labels in
the native trace do not identify the Python caller or establish initialization
as the cause. There is no Python thread traceback in the supplied crash log.

The last completed request had a 173,180-token prompt, 171,008 cached tokens,
2.335 seconds of prefill, and 104.67 tokens/s decode. Those figures show working
prefix reuse immediately before the crash. The previous prompt-budget change
preserves input that fits and limits output to the remaining capacity; it
contains no native-tokenizer changes. Its timing remains relevant because it
allows longer histories to reach later operations. It does not prove or rule
out a regression.

## Confirmed issue and patch

ExLlamaV3's `encode_part_base` mutated the shared native tokenizer's
`encode_special_tokens` flag, then encoded text without synchronization or
restoration. Python request threads can overlap token counting, prompt
encoding and generation decoding. `TOKENIZERS_PARALLELISM=false` does not
serialize those Python callers. Existing synchronization covered only some
vocabulary helpers.

The tokenizer wrapper now uses one reentrant lock per instance for mode
changes and native encode/decode/count operations, vocabulary construction,
and lazy HF template operations. The previous mode is restored in `finally`,
including when native encoding raises a Python exception. Vocabulary helpers
use the same lock to avoid introducing conflicting lock order. The lock is
released between prompt parts; no GPU prefill or generation is locked by it.
Direct third-party access to the underlying native tokenizer is outside this
wrapper's synchronization contract.

Both the source tokenizer and the installed environment copy were updated
and their SHA-256 hashes matched. Originals are backed up under
`user_data/backups/tokenizer-crash-20260909-202917/`. Reinstalling ExLlamaV3 can
overwrite the installed patch.

`server.py` enables `faulthandler` before native ML imports, writing Python
thread stacks to stderr on native faults. It does not dump prompt contents
or local variables and does not catch or recover from an access violation.
Stderr must be retained by the launcher to preserve the trace after exit.

## Validation and limits

- An initial concurrent test of the original model tokenizer did not reproduce
  the reported access violation.
- Four targeted regression tests exercise request-local special-token modes,
  restoration after exceptions, competing encode/decode/count calls, and
  concurrent output equivalence. Running them against the backed-up original
  produced five failed assertions/subtests; all pass with the patch.
- All 62 `test_exllamav3*.py` tests pass with CUDA hidden.
- The offline probe completed 16 concurrent encode/decode/count round trips
  on the actual model tokenizer with 240,137–240,143 tokens and 40 synthetic
  image embeddings. Outputs matched serial baselines. The concurrent portion
  took 5.137 seconds and `torch.cuda.is_initialized()` remained false.
  Synthetic embeddings test token handling, not the real vision encoder.
- Python compilation checks passed for the tokenizer, server and probe.

Logs are in `user_data/logs/tokenizer_safety_tests.txt`,
`tokenizer_exllamav3_regression.txt`, and `tokenizer_long_stress.txt`.

This validates the shared-state fix, not the root cause or elimination of the
reported access violation. Real request stability still needs observation
after the user restarts. No server restart, model reload, context change,
WorkBuddy budget change or dependency replacement was performed.

The test environment also reports an xFormers binary built against PyTorch
2.9.0 while this environment runs 2.10.0, plus an incompatible torchao optional
extension. The optional imports recover and tests continue. Enabling fault
traces exposed caught `0xC0000139` loader exceptions during xFormers import;
these differ from the user's runtime `0xC0000005`. They are separate dependency
maintenance findings, not established causes of this crash. No unverified
dependency upgrade/downgrade was applied.

To repeat the isolated synthetic check without calling the server:

```powershell
installer_files/env/python.exe -X utf8 scripts/probe_tokenizer_safety.py --model-dir user_data/models/Qwen3.8-27B-EXL3-3.5bpw --tokens 240000 --rounds 4
```

## Follow-up: display move and startup errors

The 20:38 startup log reported an empty active-device list during model
loading. Independent checks confirmed PyTorch saw zero GPUs and direct
`nvcuda.dll` initialization returned `CUDA_ERROR_NO_DEVICE` (100). This also
occurred outside the sandbox, although NVML (`nvidia-smi`) and Device Manager
still detected the RTX 5090 D v2. No `CUDA_VISIBLE_DEVICES` mask was present
in the diagnostic process. CUDA numbering is independent of Task Manager:
the NVIDIA card is CUDA 0 even when Task Manager calls it GPU 1.

By the subsequent check around 20:46, CUDA detection had recovered, a small
GPU calculation returned the expected result, and the live API reported the
model loaded. We did not change NVIDIA settings, restart the driver, or
restart/reload the server. The exact external change that restored CUDA was
not established.

Two startup safeguards were then staged:

- ExLlamaV3 rejects loading before allocating components when PyTorch detects
  zero CUDA devices, with an actionable error instead of an indexing failure.
- Its optional xFormers backend reads package build metadata before loading
  native libraries. This installation's PyTorch 2.9.0+cu128 build is skipped
  under PyTorch 2.10.0+cu128. Other attention backends remain available. This
  avoids the caught `0xC0000139` loader exceptions while retaining fault
  tracing for actual crashes. No packages were replaced.

The optional-backend source and installed copies match; backups are in
`user_data/backups/cuda-startup-20260909-204710/`. All 65 ExLlamaV3 tests passed
in an isolated CPU process with fault tracing enabled, without the previous
`0xC0000139` traces. Results: `user_data/logs/cuda_startup_tests.txt`.
These safeguards apply to a subsequent server restart; the live full-context
test uses the model the user already loaded.
