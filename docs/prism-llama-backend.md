# PrismML llama.cpp backend

This installation uses the official PrismML `prism` branch release
`prism-b10685-7dffb15` (commit `7dffb158d`), Windows x64 CUDA 13.3.

- Source: https://github.com/PrismML-Eng/llama.cpp/tree/prism
- Release: https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10685-7dffb15
- Installed executable: `installer_files/prism-llama/prism-b10685-7dffb15/bin/llama-server.exe`
- Both release ZIPs were checked against GitHub's SHA-256 digests before extraction.
- The executable and its matching DLLs are isolated from the stock package.

## Use

Restart textgen normally and select the **llama.cpp** loader. The persistent
`--llama-server-path` option in `user_data/CMD_FLAGS.txt` selects PrismML for
this loader. Other loaders are unaffected. The selected executable is logged
when loading a model. Do not combine this option with `--ik`.

For Ternary Bonsai 27B, place `Ternary-Bonsai-27B-PQ2_0.gguf` in
`user_data/models`, refresh the model list, and load it. Download the weights
manually from https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/tree/main.
The older legacy `Ternary-Bonsai-27B-Q2_0.gguf` is incompatible with this release;
use `PQ2_0` or `Q2_g64`. A vision projector and speculative drafter are optional.

## Roll back

Remove or comment out the `--llama-server-path` line in `user_data/CMD_FLAGS.txt`,
then restart textgen. The original `llama_cpp_binaries` package is unchanged.

## Validation

Verified release checksums, executable startup, RTX 5090 D v2 detection,
and compatibility of the main textgen launch flags. Model inference remains
unverified until weights are available.
