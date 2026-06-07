**Task:** Profile a 7-step DeepSeek-V2-Lite training run in {framework} with Nsight Systems and identify the top 3 most expensive CUDA kernels by total GPU time, aggregated across all ranks.

**Codebase:** You will be modifying the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it.

**Prerequisites:** The workspace is already setup; `train.sh` runs successfully on the baseline and can be run directly. The base model is **DeepSeek-V2-Lite**; training resumes from the released HuggingFace checkpoint (already converted to {framework}'s checkpoint format) so the model is in its trained, steady-state regime, and the DCLM corpus is staged.

- `{framework}/challenges/train.sh` — the 7-step run (steps 1–6 are warmup; step 7 is the measurement target).
- the framework source code, which you are modifying.

**Conventions:**

- Every run of `train.sh` is automatically logged to `artifacts/train-<timestamp>.log`. Read this file if you need to debug a failed run after the fact. You can safely launch it in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

1. **Profile step 7 only.** Use Nsight Systems (`nsys profile`; the `nsys` binary is available system-wide) to profile **only step 7** — steps 1–6 are warmup for cudagraph capture, NCCL handshake, and allocator priming, and including them would inflate the rankings with one-shot setup work. How you bracket the measurement window to step 7 is up to you.

2. **Extract the top-3 kernels.** From the profile, aggregate per-kernel GPU time across all ranks and take the top 3 by total GPU time, reported with demangled kernel names.

3. **Output.** Write a single CSV at `artifacts/heavy-kernels/top-kernels.csv` with header `kernel_name,total_time_ms,instances,mean_time_ms` and exactly three rows, sorted by `total_time_ms` descending. Also leave the raw report at `artifacts/heavy-kernels/profile.nsys-rep` so the result is reproducible.

**Verification workflow:**

1. Wire in the profiler (see above).
2. Run the profiling pipeline and confirm `artifacts/heavy-kernels/top-kernels.csv` (three rows) and `artifacts/heavy-kernels/profile.nsys-rep` are produced.

**Constraints:**

- The run is exactly 7 steps; profile only step 7 (steps 1–6 are warmup).
- You may modify any code inside `{framework}/` to wire in the profiler, but keep the supplied training configuration and ensure `train.sh` still completes.

**Deliverables:** A brief report covering which files you changed (check with `git -C {framework} diff --stat`) and why, and how you wired in the profiler and extracted the top-3 kernels. The committed changes must produce `top-kernels.csv` and `profile.nsys-rep` under `artifacts/heavy-kernels/` when the profiling pipeline is run. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading `{framework}/challenges/train.sh` and the framework's training entrypoint to decide where to bracket the step-7 measurement.**
