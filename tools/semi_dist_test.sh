#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG="configs/fcos_semi/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py"
GPUS=8
PORT=${PORT:-29505}

echo "the config file is: ${CONFIG}"

WORKDIR="workdir_coco/xxxx"

CHECKPOINT="${WORKDIR}/epoch_28.pth_ema"

PREFIX="${CHECKPOINT}-unlabeled"
echo "using the checkpoint file: ${CHECKPOINT}"

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval bbox





##!/usr/bin/env bash
#
#PYTHON=${PYTHON:-"python"}
#
#NUMBER=${1}
#CONFIG=${2}
#GPUS=${3}
#EPOCH=${4}
#PERCENT=${5}
#PORT=${PORT:-29503}
#
#echo "the number of exp is: ${NUMBER}, the config file is: ${CONFIG}"
#
#CHECKPOINT="./exps/${NUMBER}/epoch_${EPOCH}.pth"
#PREFIX="./data/coco/annotations/semi_supervised/${NUMBER}_${EPOCH}_instances_train2017.1@${PERCENT}-unlabeled"
#echo "using the checkpoint file: ${CHECKPOINT}"
#
#$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/test_unlabeled_data.py $CONFIG $CHECKPOINT --launcher pytorch --format-only --option "jsonfile_prefix=${PREFIX}"
#
#echo "the result is saving to :${PREFIX}.bbox.json"
#
#UNLABELED="./data/coco/annotations/semi_supervised/instances_train2017.1@${PERCENT}-unlabeled.json"
#EST="${PREFIX}.bbox.json"
#PSEUDO="./data/coco/annotations/semi_supervised/${NUMBER}_${EPOCH}_instances_train2017.1@${PERCENT}-unlabeled-pseudo.json"
#$PYTHON semi_setup/convert_output_to_pseudo_label.py ${UNLABELED} ${EST} ${PSEUDO}
