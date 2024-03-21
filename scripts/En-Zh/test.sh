#!/bin/bash

FAIRSEQ=/StyleMT/fairseq-pro-StyleMT

SRC=en
TGT=zh
DATA=/StyleMT/luxun-data/data-bin
REFDIR=/StyleMT/luxun-data
EXP=/StyleMT/En-Zh
CKPTS=$EXP/ckpts

step=best
ckpt=$CKPTS/*${step}.pt
bsz=128
alpha=2.0
seed=1
beam=4
RES=$EXP/test-res-vanilla/$step-b$beam-$alpha
mkdir -p $RES

subset=test
genout=$RES/$subset
ref=$REFDIR/$subset.luxun.random.$TGT
input=$REFDIR/$subset.luxun.random.tok.$SRC
cat $input |\
CUDA_VISIBLE_DEVICES=0 fairseq-interactive $DATA \
    -s $SRC -t $TGT \
    --gen-subset $subset \
    --path $ckpt \
    --lenpen $alpha --beam $beam \
    --buffer-size $bsz --batch-size $bsz \
    > $genout

grep ^H $genout |\
    sed 's/H-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $3}' |\
    tee $genout.spm | python zh-detok.py > $genout.detok

cat $genout.detok | sacrebleu -tok zh $ref | tee $genout.bleu
