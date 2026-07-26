#!/bin/bash
# Prepare the pith-train workspace for operate-and-profile/train-and-evaluate.

set -euo pipefail
source challenges/exports.sh

PITH_TRAIN_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/pith-train.patch)
AGENTS_NEUTRAL_PATCH=$(realpath challenges/agent-neutral.patch)
LM_EVALUATION_HARNESS_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/lm-evaluation-harness.patch)

setup_codebase()
{
    git init pith-train
    git -C pith-train remote add origin $PITH_TRAIN_URL
    git -C pith-train fetch --depth 1 origin $PITH_TRAIN_SHA
    git -C pith-train checkout FETCH_HEAD
    git -C pith-train apply $PITH_TRAIN_PATCH
    git -C pith-train apply $AGENTS_NEUTRAL_PATCH

    git init lm-evaluation-harness
    git -C lm-evaluation-harness remote add origin $LM_EVALUATION_HARNESS_URL
    git -C lm-evaluation-harness fetch --depth 1 origin $LM_EVALUATION_HARNESS_SHA
    git -C lm-evaluation-harness checkout FETCH_HEAD
    git -C lm-evaluation-harness apply $LM_EVALUATION_HARNESS_PATCH
}

build_environment()
{
    pushd pith-train
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
    git -C pith-train checkout -b main
    git -C pith-train add -A
    git -C pith-train -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"

    git -C lm-evaluation-harness checkout -b main
    git -C lm-evaluation-harness add -A
    git -C lm-evaluation-harness -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"
}

pushd $1
setup_codebase
build_environment
commit_baseline
popd
