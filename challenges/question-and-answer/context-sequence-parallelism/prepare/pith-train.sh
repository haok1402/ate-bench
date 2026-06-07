#!/bin/bash
# Prepare the pith-train workspace for question-and-answer/context-sequence-parallelism.

set -euo pipefail
source challenges/exports.sh

setup_codebase()
{
    git init pith-train
    git -C pith-train remote add origin $PITH_TRAIN_URL
    git -C pith-train fetch --depth 1 origin $PITH_TRAIN_SHA
    git -C pith-train checkout FETCH_HEAD
}

commit_baseline()
{
    git -C pith-train checkout -b main
}

pushd $1
setup_codebase
commit_baseline
popd
