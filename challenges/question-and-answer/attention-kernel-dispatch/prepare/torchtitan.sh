#!/bin/bash
# Prepare the torchtitan workspace for question-and-answer/attention-kernel-dispatch.

set -euo pipefail
source challenges/exports.sh

setup_codebase()
{
    git init torchtitan
    git -C torchtitan remote add origin $TORCHTITAN_URL
    git -C torchtitan fetch --depth 1 origin $TORCHTITAN_SHA
    git -C torchtitan checkout FETCH_HEAD
}

commit_baseline()
{
    git -C torchtitan checkout -b main
}

pushd $1
setup_codebase
commit_baseline
popd
