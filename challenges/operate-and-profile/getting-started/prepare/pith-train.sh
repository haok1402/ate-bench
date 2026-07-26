#!/bin/bash
# Prepare the pith-train workspace for operate-and-profile/getting-started.

set -euo pipefail
source challenges/exports.sh

PITH_TRAIN_PATCH=$(realpath challenges/operate-and-profile/getting-started/patches/pith-train.patch)
AGENTS_NEUTRAL_PATCH=$(realpath challenges/agent-neutral.patch)
SETUP_SCRIPT=$(realpath challenges/operate-and-profile/getting-started/prepare/setup.py)

setup_codebase()
{
    git init pith-train
    git -C pith-train remote add origin $PITH_TRAIN_URL
    git -C pith-train fetch --depth 1 origin $PITH_TRAIN_SHA
    git -C pith-train checkout FETCH_HEAD
    git -C pith-train apply $PITH_TRAIN_PATCH
    git -C pith-train apply $AGENTS_NEUTRAL_PATCH
    ln -s pith-train/AGENTS.md AGENTS.md
    ln -s pith-train/.agents .agents
}

build_environment()
{
    pushd pith-train
    uv sync
    popd
}

setup_workspace()
{
    python3 $SETUP_SCRIPT
}

discard_environment()
{
    rm -rvf pith-train/.venv pith-train/uv.lock
    rm -rf "$UV_CACHE_DIR"; mkdir -p "$UV_CACHE_DIR"
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
export PYTHONPATH=$PWD/pith-train
setup_workspace
deactivate
discard_environment
commit_baseline
popd
