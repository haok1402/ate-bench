#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# One pass per node; launch on all three nodes together for three independent sessions.
# Baseline uses the default pins; indirection overrides PithTrain to the abl/skeleton fork.

set -euo pipefail

use_baseline()
{
    unset PITH_TRAIN_URL PITH_TRAIN_SHA
    export SNAPSHOTS=snapshots
}

use_indirection()
{
    export PITH_TRAIN_URL=https://github.com/MasterJH5574/Pith-Train
    export PITH_TRAIN_SHA=a57c97a7e8303c595f9df89cc3440b9b2c5ba47a
    export SNAPSHOTS=snapshots-indirection
}

run()
{
    local challenge=$1; local agent="codex"; local model="gpt-5.6-sol"
    for framework in torchtitan pith-train Megatron-LM; do
        python3 challenges/launch.py $framework $challenge $agent $model
    done
}

run_pith()
{
    python3 challenges/launch.py pith-train $1 codex gpt-5.6-sol
}

run_megatron()
{
    python3 challenges/launch.py Megatron-LM $1 codex gpt-5.6-sol
}

# 3. indirection question-and-answer (PithTrain only)
# ----------------------------------------------------------------------------
run_pith challenges/question-and-answer/attention-kernel-dispatch
run_pith challenges/question-and-answer/configuration-propagation
run_pith challenges/question-and-answer/context-sequence-parallelism
run_pith challenges/question-and-answer/data-loading-sharding
run_pith challenges/question-and-answer/distributed-checkpoint-serialization
run_pith challenges/question-and-answer/distributed-seed-management
run_pith challenges/question-and-answer/fsdp-ddp-wrapping
run_pith challenges/question-and-answer/global-gradient-clipping
run_pith challenges/question-and-answer/normalization-placement
run_pith challenges/question-and-answer/process-groups-device-mesh
run_pith challenges/question-and-answer/rope-implementation
run_pith challenges/question-and-answer/swiglu-mlp-block
