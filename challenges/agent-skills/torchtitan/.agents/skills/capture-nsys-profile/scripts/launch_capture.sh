#!/bin/bash
# Capture an nsys profile of a single steady-state torchtitan step (DeepSeek-V2-Lite).
# torchtitan has no native cudaProfiler step bracketing, so this skill first installs a
# small hook (torch.cuda.profiler.start/stop + nvtx around the target step, coordinated
# across ranks via a CPU gloo group) into the train loop, then triggers nsys off it.

set -euo pipefail
source torchtitan/.venv/bin/activate

PROFILE_STEP=7
CONFIG=deepseek_v2_lite_report_heavy_kernels
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile-step) PROFILE_STEP=$2; shift 2 ;;
        --config) CONFIG=$2; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Install the nsys step-bracket hook into the train loop (idempotent).
HOOK=$PWD/.agents/skills/capture-nsys-profile/scripts/nsys_hook.patch
if git -C torchtitan apply --reverse --check "$HOOK" 2>/dev/null; then
    echo "nsys capture hook already installed"
else
    git -C torchtitan apply "$HOOK"
    echo "installed nsys capture hook"
fi

OUTDIR=workspace/capture-nsys-profile; mkdir -p $OUTDIR
export TORCHTITAN_NSYS_PROFILE_STEP=$PROFILE_STEP

NSYS_ARGS=()
NSYS_ARGS+=(profile)
NSYS_ARGS+=(--stats=false)
NSYS_ARGS+=(--trace=cuda,nvtx)
NSYS_ARGS+=(--force-overwrite=true)
NSYS_ARGS+=(--output=$OUTDIR/torchtitan_node0)
NSYS_ARGS+=(--cuda-graph-trace=node)
NSYS_ARGS+=(--capture-range=cudaProfilerApi)
NSYS_ARGS+=(--capture-range-end=stop)
NSYS_ARGS+=(--delay=0)

LAUNCH_ARGS=()
LAUNCH_ARGS+=(--nnodes=1 --node-rank=0 --nproc-per-node=gpu)
LAUNCH_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=localhost:15213)

export OMP_NUM_THREADS=8; export PYTHONUNBUFFERED=1; export PYTHONPATH=$PWD/torchtitan
nsys ${NSYS_ARGS[@]} torchrun ${LAUNCH_ARGS[@]} -m torchtitan.train --module deepseek_v3 --config $CONFIG
