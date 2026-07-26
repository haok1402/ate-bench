#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# Skills ablation: report-heavy-kernels on Megatron-LM + torchtitan, now with the
# capture-nsys-profile skill transplanted in. Snapshots land in snapshots-skills.
# (pith-train already has the skill on the baseline branch and is the control.)

set -euo pipefail

export SNAPSHOTS=snapshots-skills

run()
{
    local framework=$1
    python3 challenges/launch.py $framework challenges/operate-and-profile/report-heavy-kernels codex gpt-5.6-sol
}

run Megatron-LM
run torchtitan
