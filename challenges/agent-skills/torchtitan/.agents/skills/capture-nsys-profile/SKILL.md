---
name: capture-nsys-profile
description: Capture a Nsight Systems (.nsys-rep) profile of a single steady-state torchtitan training step for performance analysis. Use when the user asks to "capture an nsys profile", "profile training", "grab an nsys trace", or wants kernel timelines / MoE all-to-all overheads. Loads a released checkpoint so MoE load balancing is representative, and installs a step-bracketing CUDA-profiler hook so only the target step is recorded.
---

# Capture Nsys Profile

Capture a single-step Nsight Systems (`.nsys-rep`) trace of torchtitan training, resumed from the released DeepSeek-V2-Lite checkpoint so MoE routing is in its trained regime. Steps before the target step are warmup for CUDA-graph capture, NCCL handshake, and allocator priming; they must not be in the measured trace.

## How it works

torchtitan's built-in profiler is the **torch profiler** (`enable_profiling` / `profile_freq` / warmup / active) — it emits torch-profiler traces, not an nsys `cudaProfilerApi` capture window. So there is no native way to bracket one step for nsys. This skill supplies the missing piece:

- It patches in `maybe_enable_nsys(current_step, profile_step, sync_group)` (added to `torchtitan/tools/profiling.py`, wired into `trainer.py`). Entering the target step it calls `torch.cuda.profiler.start()` + `nvtx.range_push`; leaving it, `torch.cuda.profiler.stop()`. A CPU-only gloo group synchronizes every rank around the window without adding sync kernels to the measured GPU workload. The step is armed by the `TORCHTITAN_NSYS_PROFILE_STEP` env var.
- `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` records **only** between those triggers, so the `.nsys-rep` contains just the target step.
- `--cuda-graph-trace=node --trace=cuda,nvtx` keeps kernels attributable under CUDA graphs.

## Prerequisites

- **Env**: `source torchtitan/.venv/bin/activate`.
- **nsys CLI**: `nsys --version` works.
- **Hardware**: enough GPUs for the config's parallelism.
- **Inputs staged**: the DCP checkpoint under `checkpoints/deepseek-v2-lite/torch-dcp` and tokenized DCLM under `datasets/dclm-baseline`.

## Capture

```bash
# profile step 7 of the report-heavy-kernels config (default)
bash .agents/skills/capture-nsys-profile/scripts/launch_capture.sh

# profile a different step / config
bash .agents/skills/capture-nsys-profile/scripts/launch_capture.sh --profile-step 7 --config deepseek_v2_lite_report_heavy_kernels
```

The script installs the hook (idempotent — safe to re-run), arms `TORCHTITAN_NSYS_PROFILE_STEP`, and runs `torchrun -m torchtitan.train` under nsys.

## Output

One `.nsys-rep` at `workspace/capture-nsys-profile/torchtitan_node0.nsys-rep`, covering all local ranks. Analysis — `nsys stats`, per-kernel aggregation, top kernels — is **out of scope** for this skill; run those on the produced report separately.

## Common Issues

### Empty or tiny `.nsys-rep`
The hook never armed. Verify `TORCHTITAN_NSYS_PROFILE_STEP` is set to a step within the configured `training.steps`, and that the hook patch applied (`git -C torchtitan apply --check .agents/skills/capture-nsys-profile/scripts/nsys_hook.patch`).

### `TORCHTITAN_NSYS_PROFILE_STEP must identify a configured training step`
The requested step exceeds the config's `training.steps`. Lower `--profile-step` or raise the config's step count.

### Hook patch fails to apply
It may already be installed (the launch script checks this). If you edited the train loop, re-apply against a clean `torchtitan/` checkout.
