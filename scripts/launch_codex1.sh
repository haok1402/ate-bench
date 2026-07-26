#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# One pass per node; launch on all three nodes together for three independent sessions.

set -euo pipefail

run_pithtrain()
{
    local challenge=$1; local agent="codex"; local model="gpt-5.6-sol"
    python3 challenges/launch.py pith-train $challenge $agent $model
}

# operate-and-profile
# ----------------------------------------------------------------------------
run_pithtrain challenges/operate-and-profile/report-heavy-kernels
