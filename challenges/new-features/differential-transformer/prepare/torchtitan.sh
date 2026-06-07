#!/bin/bash
# Prepare the torchtitan workspace for new-features/differential-transformer.

set -euo pipefail
source challenges/exports.sh

TORCHTITAN_PATCH=$(realpath challenges/new-features/differential-transformer/patches/torchtitan.patch)

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
    uv sync
    popd
}

setup_workspace()
{
    CMDARG=()
    CMDARG+=(mlfoundations/dclm-baseline-1.0)
    CMDARG+=(--repo-type dataset)
    CMDARG+=(--include "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-3]_processed.jsonl.zst")
    CMDARG+=(--local-dir datasets/dclm-baseline)
    hf download ${CMDARG[@]}

    CMDARG=()
    CMDARG+=(deepseek-ai/DeepSeek-V2-Lite)
    CMDARG+=(--repo-type model)
    CMDARG+=(--include "tokenizer*")
    CMDARG+=(--local-dir checkpoints/deepseek-v2-lite)
    hf download ${CMDARG[@]}
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
setup_workspace
deactivate
commit_baseline
popd
