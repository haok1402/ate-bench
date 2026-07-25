#!/bin/bash
# native-te ablation smoke test: PithTrain routed through NVIDIA TransformerEngine.
# One task only — integrate dynamic-mixture-of-experts — PithTrain framework only.
# Snapshots land in snapshots-native to keep this isolated from baseline/indirection.

set -euo pipefail

use_native()
{
    export PITH_TRAIN_URL=https://github.com/MasterJH5574/Pith-Train
    export PITH_TRAIN_SHA=32d2cd7f503c460698afc0fba7e174aa1afc2013
    export SNAPSHOTS=snapshots-native
}

run_pith()
{
    python3 challenges/launch.py pith-train $1 codex gpt-5.6-sol
}

use_native
run_pith challenges/new-features/dynamic-mixture-of-experts
