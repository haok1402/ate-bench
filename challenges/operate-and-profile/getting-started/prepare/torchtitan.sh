#!/bin/bash
# Prepare the torchtitan workspace for operate-and-profile/getting-started.

set -euo pipefail
source challenges/exports.sh

TORCHTITAN_PATCH=$(realpath challenges/operate-and-profile/getting-started/patches/torchtitan.patch)

setup_codebase()
{
    git init torchtitan
    git -C torchtitan remote add origin $TORCHTITAN_URL
    git -C torchtitan fetch --depth 1 origin $TORCHTITAN_SHA
    git -C torchtitan checkout FETCH_HEAD
    git -C torchtitan apply $TORCHTITAN_PATCH
}

build_environment()
{
    pushd torchtitan
    uv venv
    uv pip install --python .venv/bin/python torch==2.12.0 --torch-backend=cu130
    uv pip install --python .venv/bin/python transformers zstandard
    uv sync --inexact
    popd
}

setup_workspace()
{
    CMDARG=()
    CMDARG+=(deepseek-ai/DeepSeek-V2-Lite)
    CMDARG+=(--repo-type model)
    CMDARG+=(--local-dir checkpoints/deepseek-v2-lite/hf-import)
    hf download ${CMDARG[@]}

    CMDARG=()
    CMDARG+=(mlfoundations/dclm-baseline-1.0)
    CMDARG+=(--repo-type dataset)
    CMDARG+=(--include "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-3]_processed.jsonl.zst")
    CMDARG+=(--local-dir datasets/dclm-baseline)
    hf download ${CMDARG[@]}

    CMDARG=()
    CMDARG+=(checkpoints/deepseek-v2-lite/hf-import)
    CMDARG+=(checkpoints/deepseek-v2-lite/torch-dcp/step-00000000)
    CMDARG+=(--model_name deepseek_v3 --model_flavor 16B)
    python3 torchtitan/scripts/checkpoint_conversion/convert_from_hf.py ${CMDARG[@]}
    mkdir -p checkpoints/deepseek-v2-lite/tokenizer
    cp checkpoints/deepseek-v2-lite/hf-import/tokenizer* checkpoints/deepseek-v2-lite/tokenizer/
}

discard_environment()
{
    rm -rvf torchtitan/.venv torchtitan/uv.lock
}

commit_baseline()
{
    git -C torchtitan checkout -b main
    git -C torchtitan add -A
    git -C torchtitan -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"
}

pushd $1
setup_codebase
build_environment
source torchtitan/.venv/bin/activate
export PYTHONPATH=$PWD/torchtitan
setup_workspace
deactivate
discard_environment
commit_baseline
popd
