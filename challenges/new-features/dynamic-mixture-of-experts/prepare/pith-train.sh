#!/bin/bash
# Prepare the PithTrain workspace for new-features/dynamic-mixture-of-experts.

set -euo pipefail
source challenges/exports.sh

PITH_TRAIN_PATCH=$(realpath challenges/new-features/dynamic-mixture-of-experts/patches/pith-train.patch)
AGENTS_NEUTRAL_PATCH=$(realpath challenges/agent-neutral.patch)
SETUP_SCRIPT=$(realpath challenges/new-features/dynamic-mixture-of-experts/prepare/setup.py)

setup_codebase()
{
    git init pith-train
    git -C pith-train remote add origin $PITH_TRAIN_URL
    git -C pith-train fetch --depth 1 origin $PITH_TRAIN_SHA
    git -C pith-train checkout FETCH_HEAD
    git -C pith-train apply $PITH_TRAIN_PATCH
    git -C pith-train apply $AGENTS_NEUTRAL_PATCH
}

build_environment()
{
    pushd pith-train
    uv sync
    popd
}

setup_workspace()
{
    PYTHONPATH=$PWD/pith-train python3 $SETUP_SCRIPT
}

commit_baseline()
{
    git -C pith-train checkout -b main
    git -C pith-train add -A
    git -C pith-train -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"
}

pushd $1
setup_codebase
build_environment
source pith-train/.venv/bin/activate
setup_workspace
deactivate
commit_baseline
popd
