**Task:** Integrate Mixture of Block Attention (MoBA) into {framework}

**References:**

- Paper: https://arxiv.org/abs/2502.13189
- Reference implementation: https://github.com/MoonshotAI/MoBA

**Codebase:** You will be modifying the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it.

**Prerequisites:** The workspace is already setup; both `verify.sh` and `train.sh` run successfully on the baseline and can be run directly. The base model is **DeepSeek-V2-Lite**.

- `{framework}/challenges/verify.sh` — quick smoke test (4 steps). Use this to iterate fast.
- `{framework}/challenges/train.sh` — full training run (64 steps). Use this for final validation.
- the framework source code, which you are modifying.

**Conventions:**

- Every run of `verify.sh` or `train.sh` is automatically logged to `artifacts/verify-<timestamp>.log` or `artifacts/train-<timestamp>.log`. Read these files if you need to debug a failed run after the fact. You can safely launch them in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

1. **Block partitioning and per-block keys.** Split K and V along the sequence dimension into fixed-size blocks. Compute a per-block representation as the mean of K vectors within each block. **For this challenge, use block size B = 256; with seq_len = 2048, this gives 8 blocks per sequence.**

2. **Gating with causal masking.** For each query at position t (current block `b_t = t // B`), compute gating scores `s_i = ⟨q, mean_pool(K[I_i])⟩` for every block i. Apply causal masking: future blocks (i > b_t) get `s_i = -∞`; the current block `b_t` is force-selected.

3. **Top-k routing and sparse attention.** Per query, select the top-k blocks by gating score. Compute attention only over the K, V positions belonging to those blocks; within the current block, apply standard intra-block causal masking. Aggregate via online softmax. **For this challenge, use top_k = 3 (62.5% sparsity at 8 blocks).**

**Verification workflow:**

1. Implement Mixture of Block Attention (see above).
2. Run `bash {framework}/challenges/verify.sh` after each significant change to catch crashes early.
3. Run `bash {framework}/challenges/train.sh` as final validation. Confirm training completes with finite, decreasing loss.

**Constraints:**

- Extend the existing attention path — do not rewrite it from scratch
- Follow the codebase's existing config system and code conventions
- Do not modify MLA's KV projection, RoPE, or non-attention components
- Both `verify.sh` and `train.sh` must still complete after your changes

**Deliverables:** A brief report covering which files you changed (check with `git -C {framework} diff --stat`) and why you made those changes. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading the paper and its reference implementation, then the relevant files in the `{framework}` codebase, before writing any code.**
