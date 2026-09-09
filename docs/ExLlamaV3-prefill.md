# ExLlamaV3 long-prompt prefill and disconnected streams

## Prefill workspace bounds

ExLlamaV3 allocates a job's page table for its prompt **and reserved output**. Its quantized-cache prefill staging path sizes temporary FP16 K/V buffers from the width of that table, rounded up to a power of two in pages. Passing the full table to every prompt chunk can therefore size temporary buffers for future tokens that the chunk does not read.

Textgen's job wrapper now exposes only the page-table prefix needed by the current prefill chunk. It retains every past page and all pages the chunk may write. After prefill, including failure, it restores the original table object for decoding, checkpoints and future output. It does not truncate the prompt, change the output-token limit, change KV cache precision, or release the job's allocated pages.

For illustration, with 256-token pages, four KV heads, a 256-element head dimension and two FP16 K/V buffers, a 1000-page table rounds up to a 1 GiB staging allocation. An initial 2048-token chunk needs eight pages, or 8 MiB under the same sizing formula. These are calculated workspace sizes, not measured latency gains. Later chunks still need their full past context; this does not eliminate the compute cost of a long cold prompt.

Atomic multimodal prefill is excluded because a native prefill step may expand beyond the nominal chunk to finish an image span. Native execution remains responsible for this case. MTP target/draft alignment and recurrent checkpoint boundaries remain unchanged.

## Client disconnects during first-output waits

SSE iteration runs synchronous generation in a thread pool. Cancellation of the SSE task alone can wait for an in-flight generator step to return before running the stream's cleanup. If the client disconnects during cold prefill, signalling the generation stop event only in that cleanup can leave abandoned work running.

The response wrapper now sets the request-local stop event when its existing ASGI receive loop observes http.disconnect, before SSE waits for the worker. Streaming routes inspect that event rather than starting a competing request.is_disconnected receive. This applies to completions, chat completions and Anthropic-compatible messages.

Generation still stops cooperatively. The change does not interrupt a GPU kernel already executing, kill a process, change model settings, or reload a model. It prevents a disconnected client from waiting for its first visible output merely to signal cancellation.

## Validation and activation

Run the isolated tests with the project's Python interpreter:

    python -X utf8 -m unittest discover -s tests -p test_exllamav3_prefill.py
    python -X utf8 -m unittest discover -s tests -p test_streaming_cancellation.py
    python -X utf8 -m unittest discover -s tests -p test_api_streaming_routes.py

Prefill tests compare native job execution with and without the wrapper using CPU-only model stand-ins, including cache writes, progress, MTP hidden inputs and recurrent boundaries. Streaming tests use in-process ASGI messages and stand-in generation: no real inference requests, GPU benchmarks or live-server mutations are required.

Source changes take effect when textgen next imports these modules. Editing the files does not hot-patch an already running process. A running service can be left untouched and the fix activated at a user-chosen subsequent start; do not claim a live speedup without measuring real requests after activation.
