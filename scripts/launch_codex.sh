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

# 4. baseline operate-and-profile (all frameworks)
# ----------------------------------------------------------------------------
use_baseline
run challenges/operate-and-profile/collect-routing-trace
run challenges/operate-and-profile/getting-started
run challenges/operate-and-profile/report-heavy-kernels
run challenges/operate-and-profile/train-and-evaluate
