**Task:** Integrate MoE++ into {framework}

**References:**

- Paper: https://arxiv.org/abs/2410.07348
- Reference implementation: https://github.com/SkyworkAI/MoE-plus-plus

**Codebase:** You will be modifying the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it.

**Prerequisites:** The workspace is already setup; both `verify.sh` and `train.sh` run successfully on the baseline and can be run directly. The base model is **DeepSeek-V2-Lite**.

- `{framework}/challenges/verify.sh` — quick smoke test (4 steps). Use this to iterate fast.
- `{framework}/challenges/train.sh` — full training run (64 steps). Use this for final validation.
- the framework source code, which you are modifying.

**Conventions:**

- Every run of `verify.sh` or `train.sh` is automatically logged to `artifacts/verify-<timestamp>.log` or `artifacts/train-<timestamp>.log`. Read these files if you need to debug a failed run after the fact. You can safely launch them in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

1. **Zero-computation experts (Eq. 3, 4, 5).** Add three new expert types to the existing MoE layer:
   - Zero expert: returns a zero tensor
   - Copy expert: returns the input unchanged
   - Constant expert: returns `α₁x + α₂v` where `[α₁, α₂] = softmax(Wc @ x)` and `v` is a learned vector

   The existing top-k router selects experts as before — it does not know expert types. Make it configurable which expert indices are which type. **For this challenge, use 1 zero expert, 1 copy expert, 2 constant experts, and 60 FFN experts (64 total).**

2. **Gating residuals (Eq. 6).** Each MoE layer's router computes logits as `W @ x + W_g @ prev_routing_scores`, where `prev_routing_scores` comes from the previous MoE layer. The first MoE layer uses zeros for `prev_routing_scores`.

3. **Heterogeneous load balance loss (Eq. 7, 8).** Modify the load balance loss so that zero-computation experts are weighted by a configurable hyperparameter τ. Implement heterogeneous expert capacity so FFN and zero-computation experts receive different token budgets per Eq. 8. **For this challenge, use τ = 0.75.**

**Verification workflow:**

1. Implement MoE++ (see above).
2. Run `bash {framework}/challenges/verify.sh` after each significant change to catch crashes early.
3. Run `bash {framework}/challenges/train.sh` as final validation. Confirm training completes with finite, decreasing loss.

**Constraints:**

- Extend the existing MoE implementation — do not rewrite it from scratch
- Follow the codebase's existing config system and code conventions
- Do not modify attention, embedding, or non-MoE components
- Both `verify.sh` and `train.sh` must still complete after your changes

**Deliverables:** A brief report covering which files you changed (check with `git -C {framework} diff --stat`) and why you made those changes. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading the paper and its reference implementation, then the relevant files in the `{framework}` codebase, before writing any code.**
