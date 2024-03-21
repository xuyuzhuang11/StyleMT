train=train.token

fairseq-preprocess -s en -t zh \
        --joined-dictionary \
        --trainpref $train \
        --destdir create-dict-data-bin \
        --workers 128
