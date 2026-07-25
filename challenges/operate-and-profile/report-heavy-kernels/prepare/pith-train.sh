#!/bin/bash
# Prepare the pith-train workspace for operate-and-profile/report-heavy-kernels.

set -euo pipefail
source challenges/exports.sh

PITH_TRAIN_PATCH=$(realpath challenges/operate-and-profile/report-heavy-kernels/patches/pith-train.patch)
SETUP_SCRIPT=$(realpath challenges/operate-and-profile/report-heavy-kernels/prepare/setup.py)

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
    pushd pith-train
    uv sync --python 3.12
    popd
}

setup_workspace()
{
    python3 $SETUP_SCRIPT
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
commit_baseline
popd
