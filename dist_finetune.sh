#!/usr/bin/env bash

CFG=$1
PRETRAIN=$2
GPUS=$3
PY_ARGS=${@:4}
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    main_finetune.py \
    --cfg ${CFG} --pretrained ${PRETRAIN} --launcher="pytorch" ${PY_ARGS}
