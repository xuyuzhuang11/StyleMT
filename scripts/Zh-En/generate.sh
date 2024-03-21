#!/bin/bash

FAIRSEQ=/StyleMT/fairseq-pro-StyleMT
DETOK=/StyleMT/mosesdecoder-master/scripts/tokenizer/detokenizer.perl

SRC=zh
TGT=en
DATA=/StyleMT/Zh-En/data/data-bin
REFDIR=/StyleMT/Zh-En/data
EXP=/StyleMT/Zh-En/
CKPTS=$EXP/ckpts

step=last
ckpt=$CKPTS/*${step}.pt
bsz=128
alpha=1.0
seed=1
beam=4
RES=$EXP/res-vanilla/$step-b$beam-$alpha
mkdir -p $RES

subset=test
genout=$RES/$subset
ref=$REFDIR/$subset.$TGT
input=$REFDIR/$subset.token.$SRC
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
    tee $genout.bpe |\
    sed -r 's/(@@ )|(@@ ?$)//g' | tee $genout.tok |\
    perl $DETOK -l $TGT > $genout.detok
