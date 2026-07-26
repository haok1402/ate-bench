#!/bin/bash
# Baseline operate-and-profile — node 0 pass (balanced, 10 runs).
# Full sweep on collect-routing-trace, report-heavy-kernels, and train-and-evaluate;
# getting-started only Megatron here (its torchtitan/pith-train are on nodes 1 and 2).

set -euo pipefail

export SNAPSHOTS=snapshots

run()
{
    local challenge=$1
    for framework in torchtitan pith-train Megatron-LM; do
        python3 challenges/launch.py $framework $challenge codex gpt-5.6-sol
    done
}

run_megatron()
{
    python3 challenges/launch.py Megatron-LM $1 codex gpt-5.6-sol
}

run          challenges/operate-and-profile/collect-routing-trace
run          challenges/operate-and-profile/report-heavy-kernels
run_megatron challenges/operate-and-profile/getting-started
run          challenges/operate-and-profile/train-and-evaluate
