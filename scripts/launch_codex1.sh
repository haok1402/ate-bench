#!/bin/bash
# Baseline operate-and-profile — node 1 pass (balanced, 10 runs).
# Full sweep on collect-routing-trace and train-and-evaluate; report-heavy-kernels only
# pith-train + Megatron here (torchtitan is on nodes 0 and 2); getting-started only
# torchtitan + pith-train here (Megatron is on node 0).

set -euo pipefail

export SNAPSHOTS=snapshots

run()
{
    local challenge=$1
    for framework in torchtitan pith-train Megatron-LM; do
        python3 challenges/launch.py $framework $challenge codex gpt-5.6-sol
    done
}

run_torchtitan()
{
    python3 challenges/launch.py torchtitan $1 codex gpt-5.6-sol
}

run_pith()
{
    python3 challenges/launch.py pith-train $1 codex gpt-5.6-sol
}

run_megatron()
{
    python3 challenges/launch.py Megatron-LM $1 codex gpt-5.6-sol
}

run            challenges/operate-and-profile/collect-routing-trace
run_pith       challenges/operate-and-profile/report-heavy-kernels
run_megatron   challenges/operate-and-profile/report-heavy-kernels
run_torchtitan challenges/operate-and-profile/getting-started
run_pith       challenges/operate-and-profile/getting-started
run            challenges/operate-and-profile/train-and-evaluate
