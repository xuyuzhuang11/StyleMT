#!/bin/bash
# the function of this script is to build the model checkpoint containing adapter layers using the pre-trained NMT model checkpoint
# be sure that the base model is frozen and not be trained in this step

FAIRSEQ=/StyleMT/fairseq-pro-StyleMT
SRC=en
TGT=zh
# LOADCKPT=/StyleMT/En-Zh/ckpts/checkpoint_last.pt
DATA=/StyleMT/datastore/luxun-data/data-bin
EXP=.
CKPTS=$EXP/ckpts                    # note that this ckpt is the base En-Zh or Zh-En model
LOGS=$EXP/logs
mkdir -p $CKPTS
mkdir -p $LOGS

CUDA_VISIBLE_DEVICES=3 python3 $FAIRSEQ/fairseq_cli/train.py $DATA \
    --fp16 \
    -s $SRC -t $TGT \
    --lr-scheduler inverse_sqrt --lr 0.00001 \
    --warmup-init-lr 1e-07 --warmup-updates 1000 \
    --max-update 50000 \
    --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 --attention-dropout 0.0 --relu-dropout 0.0 \
    --max-tokens 512 --update-freq 1 \
    --arch transformer_retrieval \
    --task translation_retrieval \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CKPTS \
    --tensorboard-logdir $LOGS \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --no-progress-bar --log-format simple --log-interval 10 \
    --validate-interval 1 --save-interval 1 \
    --save-interval-updates 1 --keep-interval-updates 1 --keep-last-epochs 5 \
    |& tee $LOGS/train.log
