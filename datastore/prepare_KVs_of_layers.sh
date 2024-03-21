#!/bin/bash

# usage: bash prepare_KVs_of_layers.sh

python extract_subsents_of_layers.py ./train.luxun.all.multi.zh 3 4 5 7 8 10

bash prepare_triple_corpus.sh luxun zh1 en zh2 Zh-En En-Zh all-10 0 2.0 4 128

python build_datastore.py luxun all-10 mobi-all10

# rsync --delete-before -d --progress --stats empty/ tmp/