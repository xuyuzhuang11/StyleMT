#!/bin/bash

FAIRSEQ=/StyleMT/fairseq-pro-StyleMT
SRC=en
TGT=zh
DATA=/StyleMT/luxun-data/data-bin
EXP=.
CKPTS=$EXP/tune-ckpts-t2-all10-kv-T0.1
LOGS=$EXP/logs-t2-all10-kv-T0.1
mkdir -p $CKPTS
mkdir -p $LOGS
datastore_path=/StyleMT/datastore/luxun-datastore/mobi-all10

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 $FAIRSEQ/fairseq_cli/train.py $DATA \
    --fp16 \
    -s $SRC -t $TGT \
    --arch transformer_retrieval \
    --task translation_retrieval \
    --load-datastore-from $datastore_path \
    --lr-scheduler inverse_sqrt --lr 0.0002 \
    --warmup-init-lr 1e-07 --warmup-updates 100 \
    --max-update 50000 \
    --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 --attention-dropout 0.0 --relu-dropout 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CKPTS \
    --tensorboard-logdir $LOGS \
    --criterion consistent_label_smoothed_cross_entropy \
    --label-smoothing 0.1 --inter-dropout-alpha 5 --p-inter-dropout 0.1 --kl-warmup 100 \
    --no-progress-bar --log-format simple --log-interval 10 \
    --validate-interval 1 --save-interval 1 \
    --save-interval-updates 1000 --keep-interval-updates 100 --keep-last-epochs 5 \
    |& tee -a $LOGS/train.log
