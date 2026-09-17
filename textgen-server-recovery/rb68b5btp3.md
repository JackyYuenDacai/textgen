# Textgen premature completion: server-only fix

Updated: 2026-09-17  
Repository: [JackyYuenDacai/textgen](https://github.com/JackyYuenDacai/textgen)  
Base commit: [`ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9`](https://github.com/JackyYuenDacai/textgen/tree/ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9), “strengthen the completion”  
Status: implemented and tested in a local checkout; not pushed to GitHub or deployed to the user's running server.

## Decision and scope

**The client cannot be patched. All new recovery behavior is implemented in the server.** This supersedes the earlier recommendation to add a completion gate to the client.

Textgen now stages tool-enabled generation turns and performs bounded internal repair before returning an accepted response. The existing client continues to execute ordinary function calls and submit their results. No client API change, server-side tool execution, or external agent framework is required.

This fixes concrete malformed-call handling and catches conservative patterns of premature progress-only replies. It does not guarantee that an arbitrary natural-language final answer has completed every requested deliverable.

## 1. Original findings

### Malformed calls could silently become successful answers

At the inspected base commit, [generation.py](https://github.com/JackyYuenDacai/textgen/blob/ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9/modules/api/generation.py#L299-L324) discarded malformed tool output and hid its markup. A normal stop could then become a completed response with only a prose prefix—or no visible content at all.

The following outcomes were reproduced from the original parser and extracted repository logic:

| Original model output | Original client-visible result |
| --- | --- |
| `Fetching: <tool_call>{` | `Fetching: `, no call, completed response |
| A tagged call naming an undeclared tool | Prose prefix, no call, completed response |
| One valid Qwen call followed by an unfinished second call | Entire batch discarded, empty completed response |
| `I will now process the remaining batches.` | The promise itself, no call, completed response |

The first three are protocol-handling failures. The fourth is a valid model turn that may end the task prematurely.

### Single-tool mode omitted continuation guidance

In [responses.py](https://github.com/JackyYuenDacai/textgen/blob/ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9/modules/api/responses.py#L464-L483), the long “Continue working…” instruction was in an `elif` branch. With `parallel_tool_calls=false`, only the single-call restriction was inserted. The patch makes both instructions apply in single-tool mode.

### A response boundary is not a task boundary

The [Responses route](https://github.com/JackyYuenDacai/textgen/blob/ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9/modules/api/script.py#L431-L470) runs an individual generation turn. `response.completed` means that response finished; it does not certify that the user's entire task is complete. A completed response can contain calls awaiting execution by the client.

If the client already performs multiple tool rounds, its basic tool loop exists. The server fix improves what that existing loop receives instead of requiring client modifications.

### Token limits and hosted-model differences

The inspected Responses default is already 131,072 output tokens, constrained by the loaded context and explicit request settings. Increasing that ceiling does not prevent an early normal EOS or recover a swallowed call. Disabling EOS is not a task-completion check.

Hosted models may produce more reliable tool syntax and sustain multi-step work better. The reported comparison does not establish that hosted API servers contain a hidden agent loop. The exact cause of any particular stopped session still requires its logs and response output.

## 2. Implemented server behavior

The new `modules/api/generation_recovery.py` wraps the existing generation implementation. It applies when tools are provided and tool selection is not disabled, including Responses and the shared Chat Completions generation path. Ordinary no-tool generation and prompt-only requests bypass recovery.

### Request flow

1. Generate an initial candidate using the normal backend.
2. Stage its events privately rather than delivering them immediately.
3. Inspect attempted tool markup before it is hidden by formatting.
4. Validate the complete parsed call batch, including Responses strict schemas and custom-tool formats.
5. If no calls exist, check for a short progress-only promise or empty visible output.
6. If invalid and budget remains, generate a replacement using temporary correction context.
7. Expose only the accepted candidate, or return a truthful incomplete/failed result when recovery cannot succeed.

All generation remains inside the existing request. There is no recursive HTTP request and no execution of the client's tools on the server.

### Bounded repair

| Control | Implemented behavior |
| --- | --- |
| Attempt limit | At most two additional generations after the initial candidate |
| Repeated failure | Stops when the same error and trimmed raw output recur |
| Output budget | Initial generation and repairs share the output-token allowance |
| Presets | Recovery ceiling is enforced after preset and auto-token processing |
| Backend token limit | Returns incomplete instead of retrying beyond the limit |
| Cancellation | Checks cancellation before and after attempts; existing backend cancellation remains active |
| Backend exception | Propagates as a backend failure rather than being mistaken for a repairable format error |
| Exhaustion | Reports a structured model-output failure; does not silently claim success |

The patch does not add a new wall-clock timer. Existing request cancellation applies, and generation is bounded by attempt count and output tokens. If the backend hangs, the patch is not a replacement for a backend watchdog.

### What triggers repair

- Unparseable attempted tool markup without an accepted call.
- An invalid or unfinished member of a Qwen `<tool_call>` batch.
- Calls violating the single-call policy.
- Invalid Responses strict-schema arguments or custom-tool input formats.
- A short, recognizable first-person action promise with no call, such as “I will now fetch the remaining articles.”
- Empty visible output without an executable call.

The progress detector is deliberately narrow. It supports several English and Chinese action-promise patterns, excludes questions, code, multiline responses, and common blocker statements, and is not a general semantic verifier. Unrecognized premature prose can still pass through.

The added whole-batch malformed-markup check is strongest for Qwen's `<tool_call>` format. Other known markers without any parsed call are detected, but not every model-specific mixed valid/invalid batch format is exhaustively validated.

### Temporary correction context

Repairs retain the original canonical conversation and add only the current rejected candidate plus a correction message. The message explains that no calls from that attempt were delivered or executed, requests a complete corrected batch, and allows a genuine blocker or clarification.

Discarded attempts and internal correction messages are not stored in public response history. The original request's items are not mutated. No synthetic tool results are created.

### Streaming and duplicate execution

Tool-enabled turns are buffered until the candidate is validated. A malformed first attempt cannot leak a tool call that the client executes before the server retries it. Single-tool mode also consumes the full candidate to detect extra calls before delivery.

For an accepted turn, normal API event and response formats are preserved. Failed candidate events are discarded, and the Responses route emits one terminal event. Its existing heartbeat mechanism can continue while generation/repair runs.

**Tradeoff:** tool-enabled text and reasoning are no longer delivered token by token as the backend generates them; they are released after validation. This increases time to visible output. No-tool requests retain their original streaming behavior.

This prevents duplicate execution caused by internal retries before delivery. It does not promise exactly-once execution across network retries or separate requests.

### Usage and observability

Successful or incomplete responses aggregate prompt and completion usage across all attempts, including discarded generations. Cached prompt token counts are aggregated when present. Public output contains only the accepted or incomplete candidate, so billed/consumed tokens can exceed visible output length.

Retry logs include the repair number, remaining output budget, and failure reason. Terminal repair failures use the existing error path; the patch does not add a new public usage field to failed responses.

## 3. Files changed

| File | Purpose |
| --- | --- |
| `modules/api/generation_recovery.py` | New bounded recovery coordinator, malformed-output detection, progress heuristic, validation, and usage aggregation |
| `modules/api/generation.py` | Existing generation becomes a one-attempt implementation; public wrapper adds recovery; captures raw diagnostics and enforces remaining output budget |
| `modules/api/responses.py` | Applies continuation guidance in single-tool mode and supplies validation metadata for pre-delivery checks |
| `tests/test_generation_recovery.py` | New GPU-free coordinator tests |
| `tests/test_generation_tool_streaming.py` | Updated malformed-output expectations and new generation/HTTP repair tests |
| `tests/test_responses_api.py` | Adjusts the backend-close test for full-turn validation and potentially intervening text chunks |

No client files are changed.

## 4. Configuration

These are new server environment variables introduced by this patch. Set them before starting textgen and restart the server after changing them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TEXTGEN_TOOL_REPAIR_ATTEMPTS` | `2` | Additional attempts, clamped to 0–2; 0 disables retries but retains pre-delivery validation |
| `TEXTGEN_REPAIR_PROGRESS` | `1` | Set to `0`, `false`, or `off` to disable the progress-only heuristic; malformed-call repair remains active |

Example in Windows PowerShell:

```powershell
$env:TEXTGEN_TOOL_REPAIR_ATTEMPTS = "2"
$env:TEXTGEN_REPAIR_PROGRESS = "1"
# Start textgen using your normal command in this terminal.
```

The defaults require no environment-variable changes.

## 5. Installation

The accompanying `textgen-server-recovery.zip` contains `textgen-server-recovery.patch`, this document, and installation notes. The patch is based on commit `ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9`.

Extract the archive. In a terminal inside your textgen repository, use the actual extracted patch path:

```powershell
git status --short
git apply --check "C:\path\to\textgen-server-recovery.patch"
git apply "C:\path\to\textgen-server-recovery.patch"
```

Run the second command only if `git apply --check` succeeds. If your checkout has evolved or has overlapping local edits, reconcile the patch rather than overwriting those changes. Restart textgen after applying it. The client configuration does not change.

To reverse an otherwise unchanged applied patch:

```powershell
git apply --reverse --check "C:\path\to\textgen-server-recovery.patch"
git apply --reverse "C:\path\to\textgen-server-recovery.patch"
```

## 6. Validation results

Focused tests passed:

- 16 recovery-coordinator tests.
- 17 generation/tool-streaming tests, including real in-process HTTP routes with mocked backend output.
- 3 streaming-route cancellation tests.

Total: **36 focused tests passed**. Tests cover malformed-call repair, repeated-error and attempt limits, output budgets, cancellation, strict-schema repair, single-call restrictions, progress-only repair, usage aggregation, one terminal response, and no repair-context leakage into stored history.

Commands, from the repository root:

```bash
python -m unittest discover -s tests -p test_generation_recovery.py
python -m unittest discover -s tests -p test_generation_tool_streaming.py
python -m unittest discover -s tests -p test_api_streaming_routes.py
```

The broader `test_responses_api.py` suite runs 36 tests and retains seven failing cases. The same seven failing cases were reproduced in a separate unmodified checkout of the base commit under the same dependencies. No claim is made that the full repository suite passes. Repository-pinned FastAPI 0.112.4 and sse-starlette 1.6.5 were used for final API testing.

No real GPU/model inference or live WorkBuddy/Codex session was available. Passing mocked-backend tests establishes recovery and protocol behavior, not a measured real-model success rate.

## 7. Remaining practical limits

- A short promise can be detected; an apparently substantive but actually incomplete answer may not be.
- Repair can improve malformed-call delivery but cannot give the model capabilities it lacks.
- Repair context uses additional prompt space. Very full contexts can leave insufficient room to recover.
- Two retries still can fail; the server then reports the failure honestly.
- The existing client must already execute tool calls and submit their results. No server can complete client-owned tool actions without receiving those results or separately implementing tool execution.

For the next real stopped session, inspect the finish reason, effective token/context budget, parsed and delivered calls, repair log entries, final output, and any cancellation/disconnection. These identify whether the patch addressed the observed path or whether another stopping cause remains.
