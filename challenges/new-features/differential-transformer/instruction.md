**Task:** Integrate Differential Transformer into {framework}

**References:**

- Paper: https://arxiv.org/abs/2410.05258
- Reference implementation: https://github.com/microsoft/unilm/tree/master/Diff-Transformer

**Codebase:** You will be modifying the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it.

**Prerequisites:** The workspace is already setup; both `verify.sh` and `train.sh` run successfully on the baseline and can be run directly. The base model is **DeepSeek-V2-Lite**.

- `{framework}/challenges/verify.sh` — quick smoke test (4 steps). Use this to iterate fast.
- `{framework}/challenges/train.sh` — full training run (64 steps). Use this for final validation.
- the framework source code, which you are modifying.

**Conventions:**

- Every run of `verify.sh` or `train.sh` is automatically logged to `artifacts/verify-<timestamp>.log` or `artifacts/train-<timestamp>.log`. Read these files if you need to debug a failed run after the fact. You can safely launch them in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

1. **Differential head layout (Eq. 1).** Adapt MLA's per-head QK structure so each head emits a pair `(Q1, Q2)` and `(K1, K2)`. Keep all baseline projection sizes (Q, K_nope decompression, K_rope, V, O) unchanged. Per head, split Q's nope sub-portion along `qk_nope_head_dim` into halves `(Q1_nope, Q2_nope)` and split Q's rope sub-portion along `qk_rope_head_dim` into halves `(Q1_rope, Q2_rope)`; form `Q1 = [Q1_nope | Q1_rope]` and `Q2 = [Q2_nope | Q2_rope]`, each of per-head dim `d_qk' = qk_nope_head_dim/2 + qk_rope_head_dim/2`. Apply the same construction to K. RoPE is applied to the rope sub-portion of Q and K before the size-2 split, so `Q1/Q2` (and `K1/K2`) share positional encoding. V is unchanged (per-head dim `v_head_dim`) and is shared across the two attention maps within a head — not split.

2. **Differential attention with depth-dependent λ (Eq. 2).** For each head, compute two causal softmax attention maps `A1 = softmax(Q1 K1^T / sqrt(d_qk'))` and `A2 = softmax(Q2 K2^T / sqrt(d_qk'))`, then output `head = (A1 - λ · A2) V` (V is the shared per-head value of dim `v_head_dim`). The scalar `λ` is parameterized as `λ = exp(⟨λ_q1, λ_k1⟩) - exp(⟨λ_q2, λ_k2⟩) + λ_init` with four learnable vectors `λ_q1, λ_k1, λ_q2, λ_k2 ∈ R^{{d_qk'}}` initialized from `N(0, 0.1²)`. The depth-dependent constant is `λ_init = 0.8 - 0.6 · exp(-0.3 · (l − 1))`, where `l ∈ [1, L]` is the 1-indexed layer depth and `L` is the total number of transformer layers. **For this challenge, share `λ_q*, λ_k*` across heads within a layer (one set per layer), and use the layer's own `λ_init` for that layer.**

3. **Per-head sub-norm and depth-aware scaling (Eq. 3).** After each head's differential output, apply RMSNorm over the per-head channel dimension (size `v_head_dim`), then multiply by the fixed scalar `(1 − λ_init)` (using the layer's `λ_init` constant from step 2, not the learned `λ`). Concatenate all head outputs along the channel dimension and feed the result into the existing `W^O` output projection. **For this challenge, use the same RMSNorm epsilon as the rest of the model.**

**Verification workflow:**

1. Implement Differential Transformer (see above).
2. Run `bash {framework}/challenges/verify.sh` after each significant change to catch crashes early.
3. Run `bash {framework}/challenges/train.sh` as final validation. Confirm training completes with finite, decreasing loss.

**Constraints:**

- Extend the existing attention path — do not rewrite it from scratch
- Follow the codebase's existing config system and code conventions
- Do not modify FFN, embedding, or non-attention components
- Both `verify.sh` and `train.sh` must still complete after your changes

**Deliverables:** A brief report covering which files you changed (check with `git -C {framework} diff --stat`) and why you made those changes. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading the paper and its reference implementation, then the relevant files in the `{framework}` codebase, before writing any code.**
