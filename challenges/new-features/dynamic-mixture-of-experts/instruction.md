**Task:** Integrate Dynamic Mixture of Experts (DynMoE) into {framework}

**References:**

- Paper: https://arxiv.org/abs/2405.14297
- Reference implementation: https://github.com/LINs-lab/DynMoE

**Codebase:** You will be modifying the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it.

**Prerequisites:** The workspace is already setup; both `verify.sh` and `train.sh` run successfully on the baseline and can be run directly. The base model is **DeepSeek-V2-Lite**.

- `{framework}/challenges/verify.sh` — quick smoke test (4 steps). Use this to iterate fast.
- `{framework}/challenges/train.sh` — full training run (64 steps). Use this for final validation.
- the framework source code, which you are modifying.

**Conventions:**

- Every run of `verify.sh` or `train.sh` is automatically logged to `artifacts/verify-<timestamp>.log` or `artifacts/train-<timestamp>.log`. Read these files if you need to debug a failed run after the fact. You can safely launch them in the background with stdout/stderr discarded, then wait for completion before checking the log.
- If you do watch for the run to finish, `wait` on the launched PID rather than polling with `pgrep -f` for the script or config name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the run can take quite a while to complete.**

**What to implement:**

1. **Top-any gating (Eq. 3, 4, 5, 6, 7).** Replace the top-k softmax router with the top-any mechanism. For each token `x ∈ R^d` and expert representation matrix `W_g ∈ R^{{d×K}}`:
   - Cosine similarity score `s(x) = ⟨x, W_g⟩ / (‖x‖ ‖W_g‖)`.
   - Per-expert learnable threshold `G ∈ R^K` (initialize to zero).
   - Activation indicator `g(x) = sign(σ(s(x)) − σ(G))`. `sign` is non-differentiable — bypass it with a straight-through estimator that copies the gradient of `g(x)` to `σ(s(x)) − σ(G)`.
   - Per-token activated count `k = sum(g(x))` — varies per token and may be zero during training.
   - MoE output `y = (1/k) · Σ_{{e: g(x)_e > 0}} E_e(x)` — uniform averaging across activated experts; do not weight by gating magnitudes.
   - Inference-time fallback (Eq. 7): if a token has `k = 0`, activate only the single expert with the highest `σ(s(x))`.

   **For this challenge, cap the per-token activation count at `k_max = 12` (2× the baseline top-6). When more than `k_max` experts pass the gate, retain only the top-`k_max` by gating margin `σ(s(x)) − σ(G)`.**

2. **Sparse-and-simple auxiliary loss (Eq. 8).** Replace the existing load-balance auxiliary loss with the DynMoE loss `L_aux = ‖W_g^T W_g − I_K‖² + (1/K) · Σ_e ‖w_{{g,e}}‖²`. The diversity term decorrelates expert representations; the simplicity term regularizes their norms. **For this challenge, reuse the existing aux-loss coefficient.**

3. **Adaptive expert count (§3.2, Algorithm 1).** The number of experts `K` becomes a runtime quantity that grows and shrinks during training. Over each adjustment window, record `R_E ∈ R^K` (per-expert activation counts) and `R_S ∈ R^d` (sum of token embeddings `x` with `g(x) = 0`). At each adjustment step:
   - **Remove** every expert `e` with `R_E[e] = 0` over the window — deactivate it (e.g. by masking it out of the live set, or by freeing its parameters) so it no longer routes.
   - **Add** one new expert if `R_S ≠ 0` over the window: append a new row to `W_g` initialized to `R_S / ‖R_S‖`, append `G_{{K+1}} = 0`, instantiate a fresh FFN expert `E_{{K+1}}`, and grow `K`.
   - Reset `R_E` and `R_S` after the adjustment.

   **For this challenge, start with `K_init = 16` active experts, use an adjustment interval of 16 iterations, and cap the count at `K_max = 64`.**

**Verification workflow:**

1. Implement Dynamic Mixture of Experts (see above).
2. Run `bash {framework}/challenges/verify.sh` after each significant change to catch crashes early.
3. Run `bash {framework}/challenges/train.sh` as final validation. Confirm training completes with finite, decreasing loss.

**Constraints:**

- Extend the existing MoE implementation — do not rewrite it from scratch
- Follow the codebase's existing config system and code conventions
- Do not modify attention, embedding, or non-MoE components
- Expert counts may differ across EP ranks during training (ragged EP distribution required)
- Both `verify.sh` and `train.sh` must still complete after your changes

**Deliverables:** A brief report covering which files you changed (check with `git -C {framework} diff --stat`) and why you made those changes. Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading the paper and its reference implementation, then the relevant files in the `{framework}` codebase, before writing any code.**
