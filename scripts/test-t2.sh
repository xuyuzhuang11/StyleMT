#!/bin/bash

FAIRSEQ=/StyleMT/fairseq-pro-StyleMT

SRC=en
TGT=zh
DATA=/StyleMT/datastore/luxun-data/data-bin
REFDIR=/StyleMT/datastore/luxun-data
EXP=/StyleMT/MemAug
CKPTS=/StyleMT/ckpts/tune-ckpts-t2-all10-T0.1
datastore_path=/StyleMT/datastore/luxun-datastore/mobi-all10

step=best
ckpt=$CKPTS/*${step}.pt
bsz=64
alpha=2.0
seed=1
beam=4
RES=$EXP/$step-b$beam-$alpha
mkdir -p $RES

subset=test
genout=$RES/$subset
ref=$REFDIR/$subset.luxun.random.$TGT
input=$REFDIR/$subset.luxun.random.tok.$SRC
cat $input |\
CUDA_VISIBLE_DEVICES=0 fairseq-interactive $DATA \
    --fp16 \
    --task translation_retrieval \
    --load-datastore-from $datastore_path \
    -s $SRC -t $TGT \
    --path $ckpt \
    --gen-subset $subset \
    --lenpen $alpha --beam $beam \
    --buffer-size $bsz --batch-size $bsz \
    > $genout

grep ^H $genout |\
    sed 's/H-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $3}' |\
    tee $genout.spm | python zh-detok.py > $genout.detok

cat $genout.detok | sacrebleu -tok zh $ref | tee $genout.bleu
