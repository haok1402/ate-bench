---
name: capture-nsys-profile
description: Capture a Nsight Systems (.nsys-rep) profile of a single steady-state Megatron-LM training step for performance analysis. Use when the user asks to "capture an nsys profile", "profile training", "grab an nsys trace", or wants kernel timelines / MoE all-to-all overheads. Loads a released checkpoint so MoE load balancing is representative, and uses Megatron's native step-range profiler so only the target step is recorded.
---

# Capture Nsys Profile

Capture a single-step Nsight Systems (`.nsys-rep`) trace of Megatron-LM training, resumed from the released DeepSeek-V2-Lite checkpoint (converted to Megatron torch-dist format) so MoE routing is in its trained regime. Steps before the target step are warmup for CUDA-graph capture, NCCL handshake, and allocator priming; they must not be in the measured trace.

## How it works

Megatron ships native step-range profiling:

- `--profile --profile-step-start <N-1> --profile-step-end <N> --profile-ranks 0..W-1` makes **every** rank call `cudaProfilerStart()` entering step `N` and `cudaProfilerStop()` leaving it. Profiling every rank is required for an all-rank kernel report.
- `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` records **only** between those triggers, so the `.nsys-rep` contains just step `N`.
- `--cuda-graph-trace=node --trace=cuda,nvtx` keeps kernels attributable under CUDA graphs.

This is the whole reason to profile via `cudaProfilerApi` rather than a fixed time window: the window is defined by the training loop, not a guessed delay.

## Prerequisites

- **Env**: `source Megatron-LM/.venv/bin/activate`.
- **nsys CLI**: `nsys --version` works.
- **Hardware**: enough GPUs for `world_size >= PP * EP` with `DP >= 1` (the DeepSeek-V2-Lite report config is PP=4, EP=2 → world_size 8).
- **Inputs staged**: the converted checkpoint under `checkpoints/deepseek-v2-lite/torch-dcp` and tokenized DCLM under `datasets/dclm-baseline/toktxt/deepseek-v2`.

## Capture

```bash
# profile step 7 of a 7-step run (default)
bash .agents/skills/capture-nsys-profile/scripts/launch_capture.sh

# profile a different step
bash .agents/skills/capture-nsys-profile/scripts/launch_capture.sh --profile-step 7
```

The script pins the DeepSeek-V2-Lite model/parallelism config (PP=4, EP=2, the tuned pipeline layout), runs `train-iters == profile-step` so it stops right after the profiled step, and wraps `torchrun … pretrain_gpt.py` in the nsys capture described above.

## Output

One `.nsys-rep` at `workspace/capture-nsys-profile/megatron_node0.nsys-rep`, covering all local ranks (nsys attaches to torchrun's children). Analysis — `nsys stats`, per-kernel aggregation, top kernels — is **out of scope** for this skill; run those on the produced report separately.

## Common Issues

### No `.nsys-rep` produced
Check the run log for nsys errors. Usual causes: `nsys` not on `PATH`, or `workspace/capture-nsys-profile` not writable.

### Empty or tiny `.nsys-rep`
`--capture-range=cudaProfilerApi` never triggered. Verify `--profile` and `--profile-step-start/--profile-step-end` reached `pretrain_gpt.py`, and that `profile-step-end <= train-iters`.

### `Multiple libcudart` / cuDNN load error
The node has cuda-12 alongside cuda-13; stage the empty `libcudart.so.12` shim on `LD_LIBRARY_PATH` (the script already does this) so cuDNN's fused attention loads only cu13.
