#!/bin/bash
# Prepare the torchtitan workspace for operate-and-profile/train-and-evaluate.

set -euo pipefail
source challenges/exports.sh

TORCHTITAN_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/torchtitan.patch)
LM_EVALUATION_HARNESS_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/lm-evaluation-harness.patch)

setup_codebase()
{
    git init torchtitan
    git -C torchtitan remote add origin $TORCHTITAN_URL
    git -C torchtitan fetch --depth 1 origin $TORCHTITAN_SHA
    git -C torchtitan checkout FETCH_HEAD
    git -C torchtitan apply $TORCHTITAN_PATCH

    git init lm-evaluation-harness
    git -C lm-evaluation-harness remote add origin $LM_EVALUATION_HARNESS_URL
    git -C lm-evaluation-harness fetch --depth 1 origin $LM_EVALUATION_HARNESS_SHA
    git -C lm-evaluation-harness checkout FETCH_HEAD
    git -C lm-evaluation-harness apply $LM_EVALUATION_HARNESS_PATCH
}

build_environment()
{
    pushd torchtitan
    uv sync
    popd

    pushd lm-evaluation-harness
    uv venv
    uv pip install --python .venv/bin/python setuptools wheel
    uv pip install --python .venv/bin/python -e ".[vllm]"
    uv pip install --python .venv/bin/python "vllm<0.11" "transformers>=4.55.2,<5" ray wandb matplotlib
    popd
}

commit_baseline()
{
    git -C torchtitan checkout -b main
    git -C torchtitan add -A
    git -C torchtitan -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"

    git -C lm-evaluation-harness checkout -b main
    git -C lm-evaluation-harness add -A
    git -C lm-evaluation-harness -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"
}

pushd $1
setup_codebase
build_environment
commit_baseline
popd
