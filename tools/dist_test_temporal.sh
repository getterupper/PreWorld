#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=$((RANDOM + 10000))
NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_P2P_DISABLE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_temporal.py \
    $CONFIG \
    $CHECKPOINT \
    --eval segm \
    --launcher pytorch \
    ${@:4}
