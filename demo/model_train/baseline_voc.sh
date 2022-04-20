#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
CONFIG=configs/fcos_semi/voc/r50_caffe_mslonger_tricks_0.Xdata.py
WORKDIR=workdir_voc/r50_caffe_mslonger_tricks_07data
GPU=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR
