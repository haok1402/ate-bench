#!/bin/bash
# Shared exports for the prepare scripts. The training frameworks (TorchTitan, Megatron-LM,
# PithTrain) may be overridden from the environment to ablate against a fork/SHA; the task
# and eval helpers (Megatron-Bridge, lm-evaluation-harness) stay fixed.

export TORCHTITAN_URL="${TORCHTITAN_URL:-https://github.com/pytorch/torchtitan}"
export TORCHTITAN_SHA="${TORCHTITAN_SHA:-d84e83dc4ef4615afefe32dc83c1369a50132ba3}"

export PITH_TRAIN_URL="${PITH_TRAIN_URL:-https://github.com/mlc-ai/pith-train}"
export PITH_TRAIN_SHA="${PITH_TRAIN_SHA:-23db1829a2e55596a5347a94e53a770721e79539}"

export MEGATRON_LM_URL="${MEGATRON_LM_URL:-https://github.com/NVIDIA/Megatron-LM}"
export MEGATRON_LM_SHA="${MEGATRON_LM_SHA:-3bec9aa97dda898d16ff5a89bac0ed2b6682b172}"

export MEGATRON_BRIDGE_URL=https://github.com/NVIDIA-NeMo/Megatron-Bridge
export MEGATRON_BRIDGE_SHA=9c9dd848966322fc3ed7706747ec13219ef49dda

export LM_EVALUATION_HARNESS_URL=https://github.com/EleutherAI/lm-evaluation-harness
export LM_EVALUATION_HARNESS_SHA=27988a293647d5853e48edea291640b4af54740c
