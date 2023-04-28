#!/usr/bin/env bash

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    main_mfm.py \
    --cfg ${CFG} --launcher="pytorch" ${PY_ARGS}
