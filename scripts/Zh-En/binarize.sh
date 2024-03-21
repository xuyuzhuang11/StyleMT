#!/bin/bash

train=train.token
dev=dev.token
test=test.token
dict=./data/create-dict-data-bin/dict.zh.txt

fairseq-preprocess -s zh -t en \
    --joined-dictionary --srcdict $dict \
    --trainpref $train \
    --validpref $dev \
    --testpref $test \
    --destdir data-bin \
    --workers 128
