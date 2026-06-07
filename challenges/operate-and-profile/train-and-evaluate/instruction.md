**Task:** Drive the full setup → train → export → evaluate pipeline for DeepSeek-V2-Lite using the {framework} training framework. Train **from random initialization** for 25 steps, export the resulting checkpoint to HuggingFace format, and run lm-evaluation-harness HellaSwag (zero-shot) on it via vLLM. We are testing pipeline correctness, not model quality — the HellaSwag score is expected to be near-random because 25 steps from random init produces a barely-trained model. Whatever `lm_eval` reports, that is the result.

**Codebase:** You will be working in the `{framework}` codebase. The working branch is `main`, whose committed state ("setup the workspace") is the clean baseline you start from; your changes appear as uncommitted edits on top of it. The evaluation repo `lm-evaluation-harness/` is alongside it. You may modify any code in the sandbox **except `lm-evaluation-harness/challenges/evaluate.sh`**.

**Prerequisites:** The environments are prebuilt — `{framework}/.venv` holds the framework's training dependencies, and `lm-evaluation-harness/.venv` holds `lm-eval` + `vllm` (used by the evaluation script). `uv` is available. The base model is **DeepSeek-V2-Lite**. Nothing else is staged — you download the data and drive the pipeline yourself.

- the framework source code (the `{framework}` codebase and any auxiliary repos it ships with), which you train with.
- `lm-evaluation-harness/challenges/evaluate.sh` — the provided evaluation script (HellaSwag 0-shot via vLLM on a HuggingFace-format checkpoint). **Do not modify it.**

**Data:** Train on a single DCLM shard — download only `global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst` from `mlfoundations/dclm-baseline-1.0` (25 steps need very little data, so one shard suffices). Obtain anything else the pipeline needs yourself.

**Conventions:**

- Tee training and evaluation output into `artifacts/` (e.g. `artifacts/train-<timestamp>.log`) — the same convention the provided `evaluate.sh` uses for its `artifacts/evaluate-*.log`. Read these to inspect progress or debug crashes.
- Several steps (training, evaluation) take many minutes. Launch them in the background with stdout/stderr discarded, then `wait` on the launched PID rather than polling with `pgrep -f` for the script name: that match string also appears in your own watcher's command line, so the watcher matches itself and the loop never exits. **You must never poll in an active loop to read the output because the runs can take quite a while to complete.**

**What to implement:**

1. Prepare the data: download the DCLM shard above and tokenize it into whatever format {framework} training expects. Training is from random initialization, so there is no released checkpoint to convert or weights to load.
2. Write your training script at `{framework}/challenges/train.sh` (inside the repo, so it is captured in your diff) and drive the 25-step training run end-to-end **from random initialization** (do not load the released weights), teeing combined stdout+stderr into `artifacts/train-<timestamp>.log`. Train DeepSeek-V2-Lite for **25 steps** at PP=4, EP=2, DP=1, sequence length 2048, global batch size 1024, micro batch size 1, BF16.
3. Export the resulting checkpoint to HuggingFace format under `checkpoints/deepseek-v2-lite/hf-export/` (`config.json` + safetensors). vLLM must be able to load the model standalone from that directory.
4. Run `lm-evaluation-harness/challenges/evaluate.sh` on the exported checkpoint; the HellaSwag accuracy table is at the end of stdout.

**Constraints:**

- The mesh (PP=4, EP=2, DP=1), step count (25), sequence length (2048), global batch size (1024), and BF16 precision are fixed. Initialization must be random — do not load weights from `deepseek-ai/DeepSeek-V2-Lite`. Everything else (LR, optimizer, scheduler, data preprocessing) is your choice.
- You may modify any code in the sandbox **except `lm-evaluation-harness/challenges/evaluate.sh`**.

**Deliverables:** The final HellaSwag zero-shot accuracy reported by `evaluate.sh`, plus a brief report describing the pipeline you built (which tools/scripts you invoked, any setup quirks). Before ending your final response, cancel any background harness tasks you registered: call `TaskStop` on every `Monitor` task and let any `ScheduleWakeup` timers expire by *not* re-arming them. Pending tasks block the session from exiting until each one's timeout fires, which can be tens of minutes.

**Start by reading the framework's training entrypoint and `lm-evaluation-harness/challenges/evaluate.sh` to plan the pipeline.**
