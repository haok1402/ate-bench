#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# Skills ablation: report-heavy-kernels on Megatron-LM (with the capture-nsys-profile
# skill transplanted in) -> snapshots-skills. Run on 3 nodes for n=3.

set -euo pipefail

export SNAPSHOTS=snapshots-skills

run_megatron()
{
    python3 challenges/launch.py Megatron-LM $1 codex gpt-5.6-sol
}

run_megatron challenges/operate-and-profile/report-heavy-kernels
