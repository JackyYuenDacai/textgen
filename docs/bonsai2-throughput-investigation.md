# Bonsai 2 throughput investigation ? 2026-09-19

Hardware: RTX 5090 D v2 (24 GB), Core i9-13900K, Windows. Backend: PrismML build 10685, commit `7dffb158d`, CUDA 13.3.

The tested file was `Ternary-Bonsai-2-27B-PQ2_0/Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 bytes), verified to contain `prism.hadamard.version=1`. All runs used Flash Attention and full GPU offload. No model weights, context capacity, or persistent settings were changed.

## Measurements

Native `llama-bench`: 256 generated tokens, two repetitions per case. These microbenchmarks allocate context for the tested depth, unlike the full-capacity server checks below.

| Short-context CPU threads | Decode tokens/sec |
|---|---:|
| 24 | 107.24 |
| 1 | 107.74 |
| 4 | 106.90 |
| 8 | 107.82 |

| Cache at 55,000-token depth, 8 threads | Decode tokens/sec |
|---|---:|
| q4_0 | 81.47 |
| q8_0 | 82.78 |
| f16 | 81.34 |

Actual llama-server: 247,808-token capacity, Bonsai 2 Q8_0 vision projector loaded, 55,000-token synthetic prompt, 256 generated tokens, batch/microbatch 2048/512, eight threads. Two repetitions each.

| Cache | Mean decode tokens/sec |
|---|---:|
| q4_0 | 79.48 |
| q8_0 | 80.09 |

Long-output check: 54,993-token synthetic prompt, 4,096 generated tokens, q4_0 cache, same full capacity and vision projector. One run per route.

| Route | Decode tokens/sec |
|---|---:|
| Native llama.cpp | 81.87 |
| textgen streaming /v1/completions | 79.99 |

The textgen run reused the prompt cache populated by the native run. Compare decode rates only; total elapsed time and first-token latency are not comparable. This checks plain completions, not the tool parser or recovery buffering used by WorkBuddy. Generated content was discarded.

## Conclusions and recommendations

- Keep `q4_0` KV cache at the chosen large capacity. `q8_0` gained less than 1% in the full-capacity test and costs more memory; FP16 did not improve the 55K-depth microbenchmark. FP16 was not tested at the full 247,808 capacity because of the 24 GB VRAM budget.
- Keep all layers on the GPU (65 for this 64-block model plus output), Flash Attention enabled, one slot, and MTP disabled. The tested batch/microbatch baseline is 2048/512; batch-size sweeps were not performed, so it is not claimed to be optimal.
- Thread counts from 1 to 24 differed by less than 1% at short context. Eight happened to score highest, but the difference is too small to claim a meaningful optimization.
- Expect long-context decode to be slower: about 108 tokens/sec at short context versus about 80 at 55K in these tests. No context reduction is applied or required for the measured full-capacity results.
- The previous ~48 tokens/sec trace is not an apples-to-apples baseline. At that time the inspected launch used a 214,016-token capacity with no cache quantization flags: unsupported `q2` had fallen back to FP16. VRAM pressure is a plausible contributor, not a proven sole cause. The real prompt, long tool output, Windows activity and request path also differ from these synthetic tests.
- The GPU reached roughly 575 W during benchmark processing; observed loaded clocks were around 2.67?2.82 GHz with temperatures around 67?68 C. There is no evidence here for a large gain from changing power settings. No driver, power, clock, or global environment settings were changed.
- PQ2_0 is already the publisher-preferred packing for Blackwell throughput. Built-in MTP is unavailable in this file. Speculative decoding would need a separately verified compatible drafter and an agentic prompt-cache tradeoff evaluation; none was installed.

## Runtime identity and restoration

Before testing, both the backend `/props` and textgen `/v1/internal/model/info` reported the original `Ternary-Bonsai-27B-PQ2_0` path, with 235,520-token capacity. Its file metadata identifies `Bonsai-27B`, without the Bonsai 2 Hadamard metadata. All Bonsai 2 numbers above were obtained after explicitly loading the separate Bonsai 2 file. A matched original-Bonsai microbenchmark measured 85.96 tokens/sec at 55K depth.

The original runtime/model and 235,520-token context were restored after the tests, as authorized. WorkBuddy model labels alone do not load a model in textgen: the completion endpoint ignores the request `model` field and uses the loaded model. Select/load Bonsai 2 in textgen when using its WorkBuddy profile.

## Artifacts and sources

Raw benchmark JSON, stderr logs, benchmark scripts, and restoration snapshot are under `installer_files/bonsai-speed-investigation/`.

- https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#choosing-a-packing
- https://github.com/PrismML-Eng/Bonsai-demo/blob/main/AGENTS.md
- https://github.com/PrismML-Eng/Bonsai-demo/blob/main/scripts/start_llama_server.ps1


## Performance page integration

The llama.cpp adapter now exposes native completion timings through
`get_performance_stats()`, which is consumed by both the Performance tab and the
model-info API. Restart textgen to load the updated Python adapter, then complete
a generation through textgen. No backend rebuild or model setting changes are
needed for Ternary Bonsai 2 or the Prism backend.

The history retains the latest 100 finished or interrupted requests per loaded
model instance. It reports native decode and uncached prefill rates, prefill and
decode durations, output/prompt token counts, prompt-cache reuse, and speculative
draft acceptance when supplied. Time to first output and total request time are
measured by textgen. Native milliseconds are converted to dashboard seconds;
native decode rates are preserved rather than recomputed from streamed chunks.

History updates when a request finishes or the stream closes, not for every
three-second console timing line. Reloading the model starts a new history.
Direct requests to the llama.cpp port bypass textgen and are not recorded.
GPU allocation and image-cache statistics are unavailable from this response
format and remain blank/N/A. Stopped requests without backend timings likewise
do not invent throughput numbers.

Native field definitions:
[Prism completion responses](https://github.com/PrismML-Eng/llama.cpp/blob/prism/tools/server/server-task.cpp)
and [slot timings](https://github.com/PrismML-Eng/llama.cpp/blob/prism/tools/server/server-common.cpp).
