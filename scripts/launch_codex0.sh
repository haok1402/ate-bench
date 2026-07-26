#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# Skills ablation: report-heavy-kernels on torchtitan (with the capture-nsys-profile
# skill transplanted in) -> snapshots-skills. Run on 3 nodes for n=3.

set -euo pipefail

export SNAPSHOTS=snapshots-skills

run_torchtitan()
{
    python3 challenges/launch.py torchtitan $1 codex gpt-5.6-sol
}

run_torchtitan challenges/operate-and-profile/report-heavy-kernels
