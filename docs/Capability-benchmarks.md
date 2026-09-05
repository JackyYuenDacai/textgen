# Capability benchmarks

The **Benchmarks** tab runs Inspect AI against an OpenAI-compatible `/v1` endpoint.
It supports the Inspect Evals implementations of GSM8K (maths), HumanEval (Python),
and IFEval (instruction following). The provider uses Chat Completions, not Responses.

## Setup

1. Enable TextGen's API in Session, or start with `--api`. The usual base URL is
   `http://127.0.0.1:5000/v1`. Supply the actual model ID and API key if configured.
2. Expand **Evaluation environment** and click **Install / repair benchmark environment**.
   This installs `inspect-ai==0.3.263`, `openai==3.1.0`, and `inspect-evals[ifeval]==0.19.0` into a
   separate venv under `user_data/benchmarks/env`, without changing TextGen dependencies.
   IFEval also requires Git: its scorer dependency is installed from
   `josejg/instruction_following_eval` at revision `0c495b2f95155e8b10acb919ae283bfb4d5be6e2`.
   Its English sentence-tokenizer text tables are fetched from the official NLTK data
   repository at revision `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`, and stored under
   `user_data/benchmarks/cache/nltk`. The runner replaces only the dependency's
   resource-initialization helper so it respects this cache; the scoring rules are unchanged.
3. For HumanEval, install/start Docker Desktop with Linux containers. Evaluation
   uses Inspect's Docker sandbox; generated code is never intentionally run on the host.
   Inspect manages sandbox creation and cleanup. Dataset and container downloads require network access.
4. Start with 10 samples and enough output tokens for the model's reasoning. Runs
   are sequential. A remote provider can bill for these requests.

The parent TextGen server controls one benchmark worker at a time. Closing the
browser does not cancel it. **Stop benchmark** asks the worker to cancel and clean
up; synchronous dataset loading must return before cancellation takes effect.
There is no model reload, cache reset, or global stop-generation call.

## Protocol and scores

- GSM8K: zero-shot (`fewshot=0`), Inspect's numeric answer matching.
- HumanEval: one generation per problem (single-attempt test accuracy), Inspect's
  original HumanEval tests, not HumanEval+ or LiveCodeBench.
- IFEval: Inspect's strict/loose prompt and instruction metrics and its aggregate.
- A fixed seed shuffles the dataset before the sample limit is applied. An explicit
  generation seed is also sent; support/reproducibility depends on the server.
- Thinking Enabled/Disabled sends TextGen's `enable_thinking` request field.
  For other endpoints choose Server default. Reasoning extraction and final-answer
  separation depend on the endpoint; inspect individual answers when scores seem odd.
- Each run records generation settings and evaluator versions; Inspect logs include
  task/dataset metadata and complete sample transcripts. API keys are passed through
  environment variables and excluded from the saved run configuration.
- Errors stop the evaluation and are displayed separately from incorrect answers.
  Token-limit hits are counted because truncation can prevent a valid final answer.
- Scores are for this specific protocol/subset, not directly comparable to published
  leaderboard numbers. Keep model, template, thinking, and sampling configuration
  fixed during a run. The API's model ID alone cannot identify quantization or MTP settings.

Live progress refreshes every two seconds. **Refresh saved runs** lists up to 100
recent runs; select a run and a sample to inspect answers. **Export selected run**
downloads a ZIP containing configuration, summary, and Inspect logs. Reports are
stored in `user_data/benchmarks/runs`, survive restarts, and do not include the API key.
An unfinished run from a previous server is displayed as interrupted, not resumed.
Benchmark execution and reports are disabled in multi-user mode.

Inspect log files can also be opened with the dedicated environment's `inspect view`.
See https://inspect.aisi.org.uk/ and https://github.com/UKGovernmentBEIS/inspect_evals.
