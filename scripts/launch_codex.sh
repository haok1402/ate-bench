#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# One pass per node; launch on all three nodes together for three independent sessions.

set -euo pipefail

run()
{
    local challenge=$1; local agent="codex"; local model="gpt-5.6-sol"
    for framework in torchtitan pith-train Megatron-LM; do
        python3 challenges/launch.py $framework $challenge $agent $model
    done
}

run_megatron()
{
    local challenge=$1; local agent="codex"; local model="gpt-5.6-sol"
    python3 challenges/launch.py Megatron-LM $challenge $agent $model
}

# operate-and-profile
# ----------------------------------------------------------------------------
run_megatron challenges/operate-and-profile/getting-started
