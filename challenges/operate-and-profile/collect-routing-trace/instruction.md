**Task:** Instrument DeepSeek-V2-Lite training in {framework} to dump the per-token MoE routing trace for the first 8.39M training tokens.

**Codebase:** You will be modifying the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it.

**Prerequisites:** The workspace is already setup; `train.sh` runs successfully on the baseline and can be run directly. The base model is **DeepSeek-V2-Lite**; training resumes from the released HuggingFace checkpoint (already converted to {framework}'s checkpoint format) so the router is in its trained, load-balanced regime, and the DCLM corpus is staged.

- `{framework}/challenges/train.sh` — the routing-trace run (4 steps). Use this to produce the trace.
- the framework source code, which you are modifying.

**Conventions:**

- Every run of `train.sh` is automatically logged to `artifacts/train-<timestamp>.log`. Read this file if you need to debug a failed run after the fact. You can safely launch it in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

Capture each MoE layer's routing decision — the top-k expert IDs and their gating weights — for every token in the global batch, and write one file per step under `artifacts/routing-traces/`, named `step-<step_id:08d>.npz`. Routing decisions are model-intrinsic (a deterministic function of the loaded weights and the input tokens) and are valid from step 1; no warmup is needed. Each file contains two arrays:

- `expert_ids` — `int32`, shape `(n_moe_layers, global_batch_size, sequence_length, top_k)`. Each value in `[0, n_routed_experts)`. DSV2-Lite's first decoder layer is dense, so `n_moe_layers = 26` (decoder layers 1–26), ordered along the first axis by decoder layer.
- `gate_weights` — `float32`, same shape as `expert_ids`. Per-token weights along the `top_k` axis sum to ~1 within float tolerance.

The files (one per step) must be readable with `numpy.load` from a single rank.

**Verification workflow:**

1. Implement the routing-trace instrumentation (see above).
2. Run `bash {framework}/challenges/train.sh` and confirm it completes, leaving one `.npz` file per step under `artifacts/routing-traces/`.

**Constraints:**

- The run is exactly 4 steps; the trace covers all 4, totaling 1024 * 2048 * 4 = 8.39M tokens.
- You may modify any code inside `{framework}/` to support instrumentation (including compilation, fusion, or kernel-graph flags), but keep the supplied training configuration and ensure `train.sh` still completes.

**Deliverables:** A brief report covering which files you changed (check with `git -C {framework} diff --stat`) and why, and where you instrumented the routing path and any framework abstractions you had to work around. The committed changes must leave one `.npz` file per step under `artifacts/routing-traces/` when `train.sh` is run. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading the relevant files in the `{framework}` codebase before writing any code.**
