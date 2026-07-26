#!/bin/bash
# Baseline operate-and-profile — node 2 pass (balanced, 9 runs).
# Full sweep on getting-started, report-heavy-kernels, and train-and-evaluate.

set -euo pipefail

export SNAPSHOTS=snapshots

run()
{
    local challenge=$1
    for framework in torchtitan pith-train Megatron-LM; do
        python3 challenges/launch.py $framework $challenge codex gpt-5.6-sol
    done
}

run challenges/operate-and-profile/getting-started
run challenges/operate-and-profile/report-heavy-kernels
run challenges/operate-and-profile/train-and-evaluate
