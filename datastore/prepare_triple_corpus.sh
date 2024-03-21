#!/bin/bash

# usage: bash prepare_triple_corpus.sh $NAME $SRC1 $TGT1 $SRC2 $MODEL1_DIR_NAME $MODEL2_DIR_NAME $SUBSET_NAME $VISIBLE_DEVS $ALPHA $BEAM $BSZ
# eg: bash prepare_triple_corpus.sh luxun zh1 en zh2 Zh-En En-Zh all-10 0 2.0 4 128

current_dir=$(dirname $0)
NAME=$1
SRC1=$2
TGT1=$3
SRC2=$4
MODEL1_DIR_NAME=$5
MODEL2_DIR_NAME=$6
SUBSET_NAME=$7
VISIBLE_DEVS=$8
alpha=$9
beam=${10}
bsz=${11}
path=$current_dir/$SUBSET_NAME
tmp_path=$current_dir/tmp/$SUBSET_NAME

DATA=$current_dir/../$MODEL1_DIR_NAME/data/data-bin
SRC=$SRC1
TGT=$TGT1
step=best
ckpt=$current_dir/../$MODEL1_DIR_NAME/ckpts/*${step}.pt
subset=test

echo "Translating $NAME from $SRC to $TGT..."
for i in {0..5}; do
  echo $i
  input=$path/$i/$i.$SRC
  genout=$path/$i/$i.$TGT
  cat $input |\
  CUDA_VISIBLE_DEVICES=$VISIBLE_DEVS fairseq-interactive $DATA \
    -s $SRC -t $TGT \
    --gen-subset $subset \
    --path $ckpt \
    --lenpen $alpha --beam $beam \
    --buffer-size $bsz --batch-size $bsz \
    > $genout
  grep ^H $genout |\
    sed 's/H-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $3}' |\
    tee $genout.spm | python detokenize.py > $genout.detok
  python tokenize.py $genout.detok
  rm $genout
  rm $genout.spm
  rm $genout.detok
  mv $genout.detok.tok $genout
done

DATA=$current_dir/../$MODEL2_DIR_NAME/data/data-bin
SRC=$TGT1
TGT=$SRC2
ckpt=$current_dir/../$MODEL2_DIR_NAME/ckpts/*${step}.pt
echo "Translating $NAME from $SRC to $TGT..."
for i in {0..5}; do
  echo $i
  input=$path/$i/$i.$SRC
  genout=$path/$i/$i.$TGT
  cat $input |\
  CUDA_VISIBLE_DEVICES=$VISIBLE_DEVS fairseq-interactive $DATA \
    -s $SRC -t $TGT \
    --gen-subset $subset \
    --path $ckpt \
    --lenpen $alpha --beam $beam \
    --buffer-size $bsz --batch-size $bsz \
    > $genout
  grep ^H $genout |\
    sed 's/H-//g' | sort -k 1 -n -t ' ' | awk -F'\t' '{print $3}' |\
    tee $genout.spm | python detokenize.py > $genout.detok
  python tokenize.py $genout.detok
  rm $genout
  rm $genout.spm
  rm $genout.detok
  mv $genout.detok.tok $genout
done

python data_select.py $path $SRC1 $TGT1 $SRC2

for i in {0..5}; do
  genout=$path/$i/$i.$SRC1
  rm $genout
  mv $genout.out $genout
  genout=$path/$i/$i.$TGT1
  rm $genout
  mv $genout.out $genout
  genout=$path/$i/$i.$SRC2
  rm $genout
  mv $genout.out $genout
done

echo "Binarizing triple corpus from 0 to 5 layer..."
dict=$current_dir/../$MODEL2_DIR_NAME/data/create-dict-data-bin/dict.$TGT1.txt
for i in {0..5}; do
  echo $i
  test=$path/$i/$i
  fairseq-preprocess -s $TGT1 -t $SRC1 \
    --joined-dictionary --srcdict $dict \
    --testpref $test \
    --destdir $path/$i/data-bin \
    --workers 128
  fairseq-preprocess -s $TGT1 -t $SRC2 \
    --joined-dictionary --srcdict $dict \
    --testpref $test \
    --destdir $path/$i/data-bin \
    --workers 128
done

mkdir -p $tmp_path
for i in {0..5}; do
  mkdir -p $tmp_path/mono.$i
  mkdir -p $tmp_path/bili.$i
done

echo "Preparing monolingual and bilingual datastores..."
for i in {0..5}; do         # i-th layer & i-th dataset
  echo "monolingual values and bilingual keys&values of $i layer"
  CUDA_VISIBLE_DEVICES=$VISIBLE_DEVS fairseq-generate $path/$i/data-bin \
    -s $TGT1 -t $SRC1 --gen-subset test --path $ckpt \
    --lenpen $alpha --beam $beam --batch-size 1 --score-reference \
    --save-datastore-path $tmp_path \
    --target-layer $i --bilingual-enabled

  echo "monolingual keys of $i layer"
  CUDA_VISIBLE_DEVICES=$VISIBLE_DEVS fairseq-generate $path/$i/data-bin \
    -s $TGT1 -t $SRC2 --gen-subset test --path $ckpt \
    --lenpen $alpha --beam $beam --batch-size 1 --score-reference \
    --save-datastore-path $tmp_path \
    --target-layer $i
done
