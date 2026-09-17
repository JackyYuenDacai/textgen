# Textgen server-only recovery patch

Apply to JackyYuenDacai/textgen at base commit ee3ce804d4c1a5e5c1d84ea98e2b5d4eccc514c9.
No client changes are required. This has not been pushed or deployed.

From your textgen repository, use the extracted patch's actual path:

```sh
git apply --check /path/to/textgen-server-recovery.patch
git apply /path/to/textgen-server-recovery.patch
```

Apply only if the check succeeds, then restart textgen. See rb68b5btp3.md for findings, configuration, tests, tradeoffs, and limitations. This patch buffers tool-enabled turns until validation and may use two additional model generations within the shared output-token budget.
