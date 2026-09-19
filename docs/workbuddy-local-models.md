# WorkBuddy local model switching

The previous `WORKBUDDY_LOCAL_QWEN_ONLY_V1` patch pinned the model ID, default
selection, session model configuration, model list and inference URL to one Qwen
model. Changing WorkBuddy's configured model to Bonsai 2 therefore failed before
an API request was made.

`WORKBUDDY_LOCAL_MODELS_V2` replaces that policy in both installed WorkBuddy 5.5.6
CLI bundles (`codebuddy.js` and `codebuddy-headless.js`).

- Lists all enabled models with HTTP(S) loopback endpoints; no model-name allowlist.
- Resolves the session request selection, session options, environment/default
  selection and current model list each time a model is configured.
- Preserves each model's ID, URL, protocol flags, aliases, capabilities and budgets.
- Uses another enabled local configuration when a stored selection was removed.
  Reports no available local model if none remain; it never chooses a cloud fallback.
- Allows different local ports and API paths, including `/v1/responses` and
  `/v1/chat/completions`; does not rewrite bodies or force a single request method.
- Keeps inference redirects and HTTP proxies disabled, and checks the destination
  again after Axios request transforms. It does not block authentication, tools,
  telemetry or other non-inference application traffic.

## Apply and switch models

Fully exit and reopen WorkBuddy once after installation, because running CLI
processes retain the old JavaScript. Thereafter, add/edit local providers in
WorkBuddy's model configuration and use its model selector. No bundle repatching
is needed when changing model names, local ports or API paths.

Selecting a model in WorkBuddy does not itself load weights in textgen. Load the
corresponding model in textgen and use the matching WorkBuddy configuration.
Local-only means loopback (`localhost`, `127.x.x.x`, `[::1]`); LAN/cloud addresses
are not enabled by this patch.

## Maintenance and rollback

Sources: `scripts/workbuddy/local-model-policy.js` and
`scripts/workbuddy/patch_local_models.py`. Behavior checks:
`tests/test_workbuddy_local_policy.cjs`.

The patch tool validates both complete bundles before deployment, saves originals
under `backups/workbuddy-local-models-<timestamp>`, and verifies installed hashes.
To roll back, close WorkBuddy, copy both backed-up bundles to their original
`resources/app.asar.unpacked/cli/dist` directory, and reopen WorkBuddy.

WorkBuddy updates may overwrite the patch. The patcher intentionally rejects
unrecognized bundles instead of modifying unreviewed code.

Syntax and simulated model-selection/request-policy checks passed for both bundles.
A live conversational turn remains to be verified after WorkBuddy restarts.
