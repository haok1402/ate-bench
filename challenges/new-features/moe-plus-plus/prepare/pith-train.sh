#!/bin/bash
# Prepare the PithTrain workspace for new-features/moe-plus-plus.

set -euo pipefail
source challenges/exports.sh

PITH_TRAIN_PATCH=$(realpath challenges/new-features/moe-plus-plus/patches/pith-train.patch)
SETUP_SCRIPT=$(realpath challenges/new-features/moe-plus-plus/prepare/setup.py)

setup_codebase()
{
    git init pith-train
    git -C pith-train remote add origin $PITH_TRAIN_URL
    git -C pith-train fetch --depth 1 origin $PITH_TRAIN_SHA
    git -C pith-train checkout FETCH_HEAD
    git -C pith-train apply $PITH_TRAIN_PATCH
}

build_environment()
{
    # TransformerEngine 2.12 builds from source and needs NCCL/cuDNN headers on the
    # host-compile path; stage them and expose via CPATH. Pin Python to 3.12 (TE predates 3.14).
    uv pip install --target .te-build-headers nvidia-nccl-cu13 nvidia-cudnn-cu13
    export CPATH="$PWD/.te-build-headers/nvidia/nccl/include:$PWD/.te-build-headers/nvidia/cudnn/include${CPATH:+:$CPATH}"
    # TE's build_ext defaults to a single compile job, which serializes hundreds of CUDA
    # translation units across two arches. Cap at 64 to stay well inside host memory.
    export MAX_JOBS="${MAX_JOBS:-64}"
    # A prior TE build leaves a stale in-source CMakeCache in the shared uv checkout
    # (it points at a deleted ephemeral build env, so CMake aborts); wipe it so CMake
    # reconfigures cleanly. The built wheel stays cached, so this only bites on a miss.
    rm -rf "${UV_CACHE_DIR:-$HOME/.cache/uv}"/git-v0/checkouts/*/*/build
    pushd pith-train
    uv sync --python 3.12
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
