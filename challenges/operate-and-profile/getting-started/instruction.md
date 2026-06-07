**Task:** Set up the Python environment for the {framework} training framework from scratch, then run the provided 5-step smoke training. Success is `train.sh` reaching step 5 with a finite loss.

**Codebase:** You will be working in the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from. Your job is **purely environment setup** — build a virtual environment at `{framework}/.venv` with whatever dependencies the framework needs so the provided training script runs as-is. `uv` is installed and is the recommended package manager; Docker is not available on this host.

**Prerequisites:** The data and checkpoint are already staged — pre-tokenized DCLM under `datasets/` and the converted DeepSeek-V2-Lite checkpoint under `checkpoints/`, and the training config references them directly. The base model is **DeepSeek-V2-Lite**.

- `{framework}/challenges/train.sh` — the 5-step smoke run. Do **not** modify it; it encodes the intended MoE training setup. Your task is to make `bash {framework}/challenges/train.sh` succeed as-is.
- the framework source code, which you build the environment for.

**Conventions:**

- Every run of `train.sh` is automatically logged to `artifacts/train-<timestamp>.log`. Read this file if you need to debug a failed run after the fact. You can safely launch it in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

1. Build the environment at `{framework}/.venv` — install every dependency the framework needs to import and train, resolving the correct PyTorch / CUDA build, the framework's own requirements, and any compiled extensions.
2. Run `bash {framework}/challenges/train.sh`.
3. Confirm the log shows steps 1–5 with finite cross-entropy loss.

**Verification workflow:**

1. Build `{framework}/.venv`.
2. Run `bash {framework}/challenges/train.sh` and confirm it reaches step 5 with finite loss in `artifacts/train-<timestamp>.log`.

**Constraints:**

- Do not modify `train.sh`; it documents the intended training setup. Build the environment so the run succeeds as-is.
- Keep the supplied training configuration — the run is exactly 5 steps.

**Deliverables:** A working `{framework}/.venv` and a `bash {framework}/challenges/train.sh` run whose log shows steps 1–5 with finite loss. Provide a brief report describing how you built the environment and any setup quirks you encountered. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by inspecting the framework source and `{framework}/challenges/train.sh` to work out the dependencies, then build the environment.**
