#!/bin/bash
# native-te ablation smoke test: PithTrain routed through NVIDIA TransformerEngine.
# One task only — operate-and-profile/getting-started (the agent builds the native-te
# environment from scratch, guided by the fork's updated README). PithTrain framework only.
# Snapshots land in snapshots-native to keep this isolated from baseline/indirection.

set -euo pipefail

use_native()
{
    export PITH_TRAIN_URL=https://github.com/MasterJH5574/Pith-Train
    export PITH_TRAIN_SHA=82ae3e9d7a16223fa05d13c90fba304ebbae9a63
    export SNAPSHOTS=snapshots-native
}

run_pith()
{
    python3 challenges/launch.py pith-train $1 codex gpt-5.6-sol
}

use_native
run_pith challenges/operate-and-profile/getting-started
