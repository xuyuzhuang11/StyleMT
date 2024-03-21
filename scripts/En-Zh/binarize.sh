#!/bin/bash

train=train.token
dev=dev.token
test=test.token
dict=./En-Zh/data/create-dict-data-bin/dict.en.txt

fairseq-preprocess -s en -t zh \
    --joined-dictionary --srcdict $dict \
    --trainpref $train \
    --validpref $dev \
    --testpref $test \
    --destdir data-bin \
    --workers 128
