#!/bin/bash

FAIRSEQ=/StyleMT/fairseq-pro-StyleMT
SRC=en
TGT=zh
DATA=/StyleMT/En-Zh/data/data-bin
EXP=.
CKPTS=$EXP/ckpts
LOGS=$EXP/logs
mkdir -p $CKPTS
mkdir -p $LOGS

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 $FAIRSEQ/fairseq_cli/train.py $DATA \
    --fp16 \
    -s $SRC -t $TGT \
    --lr-scheduler inverse_sqrt --lr 0.0007 \
    --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --max-update 200000 \
    --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 --attention-dropout 0.0 --relu-dropout 0.0 \
    --max-tokens 8192 --update-freq 1 \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CKPTS \
    --tensorboard-logdir $LOGS \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --no-progress-bar --log-format simple --log-interval 10 \
    --validate-interval 1 --save-interval 1 \
    --save-interval-updates 1000 --keep-interval-updates 20 --keep-last-epochs 5 \
    |& tee $LOGS/train.log
