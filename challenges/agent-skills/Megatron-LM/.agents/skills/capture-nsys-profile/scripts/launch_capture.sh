#!/bin/bash
# Capture an nsys profile of a single steady-state Megatron-LM step (DeepSeek-V2-Lite),
# using Megatron's native step-range profiler as the nsys cudaProfilerApi trigger.

set -euo pipefail
source Megatron-LM/.venv/bin/activate

PROFILE_STEP=7
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile-step) PROFILE_STEP=$2; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Hide host libcudart.so.12 so cuDNN loads only cu13 (avoids "Multiple libcudart").
CUDART_SHIM=/tmp/cudart12-shim
mkdir -p $CUDART_SHIM && : > $CUDART_SHIM/libcudart.so.12
export LD_LIBRARY_PATH=$CUDART_SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

OUTDIR=workspace/capture-nsys-profile; mkdir -p $OUTDIR

declare -A MODEL_CONFIG
MODEL_CONFIG[tokenizer-type]=HuggingFaceTokenizer
MODEL_CONFIG[tokenizer-model]=checkpoints/deepseek-v2-lite/hf-import
MODEL_CONFIG[vocab-size]=102400
MODEL_CONFIG[num-layers]=27
MODEL_CONFIG[hidden-size]=2048
MODEL_CONFIG[ffn-hidden-size]=10944
MODEL_CONFIG[num-attention-heads]=16
MODEL_CONFIG[position-embedding-type]=rope
MODEL_CONFIG[rotary-base]=10000
MODEL_CONFIG[max-position-embeddings]=163840
MODEL_CONFIG[multi-latent-attention]=true
MODEL_CONFIG[qk-head-dim]=128
MODEL_CONFIG[qk-pos-emb-head-dim]=64
MODEL_CONFIG[kv-lora-rank]=512
MODEL_CONFIG[v-head-dim]=128
MODEL_CONFIG[qk-layernorm]=true
MODEL_CONFIG[normalization]=RMSNorm
MODEL_CONFIG[norm-epsilon]=1e-6
MODEL_CONFIG[attention-dropout]=0.0
MODEL_CONFIG[hidden-dropout]=0.0
MODEL_CONFIG[disable-bias-linear]=true
MODEL_CONFIG[swiglu]=true
MODEL_CONFIG[untie-embeddings-and-output-weights]=true
MODEL_CONFIG[enable-experimental]=true
MODEL_CONFIG[num-experts]=64
MODEL_CONFIG[moe-layer-freq]="([0]+[1]*26)"
MODEL_CONFIG[moe-ffn-hidden-size]=1408
MODEL_CONFIG[moe-shared-expert-intermediate-size]=$((1408 * 2))
MODEL_CONFIG[moe-router-dtype]=fp32
MODEL_CONFIG[moe-router-score-function]=softmax
MODEL_CONFIG[moe-router-topk]=6
MODEL_CONFIG[moe-router-load-balancing-type]=seq_aux_loss
MODEL_CONFIG[moe-aux-loss-coeff]=1e-2

declare -A INFRA_CONFIG
INFRA_CONFIG[bf16]=true
INFRA_CONFIG[transformer-impl]=transformer_engine
INFRA_CONFIG[expert-model-parallel-size]=2
INFRA_CONFIG[pipeline-model-parallel-size]=4
INFRA_CONFIG[pipeline-model-parallel-layout]="Et*4|t*4|t*4|t*3|t*3|t*3|t*3|t*3L"
INFRA_CONFIG[moe-grouped-gemm]=true
INFRA_CONFIG[moe-router-fusion]=true
INFRA_CONFIG[moe-permute-fusion]=true
INFRA_CONFIG[moe-shared-expert-overlap]=true
INFRA_CONFIG[moe-token-dispatcher-type]=alltoall
INFRA_CONFIG[use-distributed-optimizer]=true
INFRA_CONFIG[overlap-param-gather]=true
INFRA_CONFIG[overlap-grad-reduce]=true
INFRA_CONFIG[cross-entropy-loss-fusion]=true
INFRA_CONFIG[cross-entropy-fusion-impl]=te
INFRA_CONFIG[no-create-attention-mask-in-dataloader]=true
INFRA_CONFIG[no-check-for-nan-in-loss-and-grad]=true
INFRA_CONFIG[manual-gc]=true
INFRA_CONFIG[manual-gc-interval]=10
INFRA_CONFIG[cuda-graph-impl]=transformer_engine
INFRA_CONFIG[cuda-graph-scope]="attn moe_router moe_preprocess"
INFRA_CONFIG[te-rng-tracker]=true

declare -A TRAIN_CONFIG
TRAIN_CONFIG[load]=checkpoints/deepseek-v2-lite/torch-dcp
TRAIN_CONFIG[no-load-optim]=true
TRAIN_CONFIG[no-load-rng]=true
TRAIN_CONFIG[train-iters]=$PROFILE_STEP
TRAIN_CONFIG[micro-batch-size]=1
TRAIN_CONFIG[global-batch-size]=1024
TRAIN_CONFIG[seq-length]=2048
TRAIN_CONFIG[lr]=1e-6
TRAIN_CONFIG[lr-decay-style]=constant
TRAIN_CONFIG[init-method-std]=0.02
TRAIN_CONFIG[optimizer]=adam
TRAIN_CONFIG[eval-iters]=0
TRAIN_CONFIG[eval-interval]=$PROFILE_STEP
TRAIN_CONFIG[log-interval]=1
TRAIN_CONFIG[log-throughput]=true

DATA_ARGS_PATH=datasets/data_args.txt
: > $DATA_ARGS_PATH
find datasets/dclm-baseline/toktxt/deepseek-v2/ -type f -name "*.idx" | sort | while read -r FILE; do
    printf "1.0 %s " ${FILE%.idx} >> $DATA_ARGS_PATH
done
TRAIN_CONFIG[data-args-path]=$DATA_ARGS_PATH
TRAIN_CONFIG[split]=1000,0,0

# Native step-range profiling: bracket only the profiled step, on every rank.
WORLD=$(nvidia-smi -L | wc -l)
declare -A PROFILE_CONFIG
PROFILE_CONFIG[profile]=true
PROFILE_CONFIG[profile-step-start]=$((PROFILE_STEP - 1))
PROFILE_CONFIG[profile-step-end]=$PROFILE_STEP

MAIN_ARGS=()
for cfg in MODEL_CONFIG TRAIN_CONFIG INFRA_CONFIG PROFILE_CONFIG; do
    declare -n ref=$cfg
    for key in ${!ref[@]}; do
        val=${ref[$key]}
        case $val in true) MAIN_ARGS+=(--$key) ;; false) ;; *) MAIN_ARGS+=(--$key $val) ;; esac
    done
done
MAIN_ARGS+=(--profile-ranks $(seq 0 $((WORLD - 1))))

NSYS_ARGS=()
NSYS_ARGS+=(profile)
NSYS_ARGS+=(--stats=false)
NSYS_ARGS+=(--trace=cuda,nvtx)
NSYS_ARGS+=(--force-overwrite=true)
NSYS_ARGS+=(--output=$OUTDIR/megatron_node0)
NSYS_ARGS+=(--cuda-graph-trace=node)
NSYS_ARGS+=(--capture-range=cudaProfilerApi)
NSYS_ARGS+=(--capture-range-end=stop)
NSYS_ARGS+=(--delay=0)

LAUNCH_ARGS=()
LAUNCH_ARGS+=(--nnodes=1 --node-rank=0 --nproc-per-node=gpu)
LAUNCH_ARGS+=(--rdzv-backend=c10d --rdzv-endpoint=localhost:15213)

export OMP_NUM_THREADS=8; export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/Megatron-LM:$PWD/Megatron-Bridge/src
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1; export NVTE_FUSED_ATTN=1; export NVTE_USE_CUTLASS_GROUPED_GEMM=1

nsys ${NSYS_ARGS[@]} torchrun ${LAUNCH_ARGS[@]} Megatron-LM/pretrain_gpt.py ${MAIN_ARGS[@]}
