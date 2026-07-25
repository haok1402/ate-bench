#!/bin/bash
# codex + gpt-5.6-sol at high reasoning effort.
# run <challenge> launches all three frameworks once.

set -euo

run()
{
    local challenge=$1; local agent="codex"; local model="gpt-5.6-sol"
    for framework in torchtitan pith-train Megatron-LM; do
        python3 challenges/launch.py $framework $challenge $agent $model
    done
}

# question-and-answer
# ----------------------------------------------------------------------------
# run challenges/question-and-answer/attention-kernel-dispatch
# run challenges/question-and-answer/configuration-propagation
# run challenges/question-and-answer/context-sequence-parallelism
# run challenges/question-and-answer/data-loading-sharding
# run challenges/question-and-answer/distributed-checkpoint-serialization
# run challenges/question-and-answer/distributed-seed-management
# run challenges/question-and-answer/fsdp-ddp-wrapping
# run challenges/question-and-answer/global-gradient-clipping
# run challenges/question-and-answer/normalization-placement
# run challenges/question-and-answer/process-groups-device-mesh
# run challenges/question-and-answer/rope-implementation
# run challenges/question-and-answer/swiglu-mlp-block

# operate-and-profile
# ----------------------------------------------------------------------------
# run challenges/operate-and-profile/collect-routing-trace
# run challenges/operate-and-profile/getting-started
# run challenges/operate-and-profile/report-heavy-kernels
# run challenges/operate-and-profile/train-and-evaluate

# new-features
# ----------------------------------------------------------------------------
# run challenges/new-features/differential-transformer
# run challenges/new-features/dynamic-mixture-of-experts
# run challenges/new-features/mixture-of-block-attention
# run challenges/new-features/moe-plus-plus
