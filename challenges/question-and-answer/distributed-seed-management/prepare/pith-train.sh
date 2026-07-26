#!/bin/bash
# Prepare the pith-train workspace for question-and-answer/distributed-seed-management.

set -euo pipefail
source challenges/exports.sh

AGENTS_NEUTRAL_PATCH=$(realpath challenges/agent-neutral.patch)

setup_codebase()
{
    git init pith-train
    git -C pith-train remote add origin $PITH_TRAIN_URL
    git -C pith-train fetch --depth 1 origin $PITH_TRAIN_SHA
    git -C pith-train checkout FETCH_HEAD
    git -C pith-train apply $AGENTS_NEUTRAL_PATCH
}

commit_baseline()
{
    git -C pith-train checkout -b main
}

pushd $1
setup_codebase
commit_baseline
popd
