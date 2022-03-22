#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/ablation_pretrain_data


TIME=317_gamma0.3
OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/Ablation_pretrain/${TIME}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

for prefix in  'AA' 'AB' 'AC' 'AD' 'AE' 'AF' 'AG' 'AH' 'AI' 'AJ' 'AK' 'AL' 'AM'
# for prefix in 'AH' 'AI' 'AJ' 'AK' 'AL' 'AM'
do
CUDA_VISIBLE_DEVICES=5   python -u /home/ljh/CSC/Enhanced_Syllable_Feature/ablation/the_whole_pinyin_label/pretrain/train.py \
  --bert_path $BERT_PATH \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --max_epoch=1 \
  --lr=1e-4 \
  --warmup_proporation 0.1 \
  --batch_size=16 \
  --gpus=0, \
  --strategy ddp \
  --workers 8 \
  --accumulate_grad_batches 32 \
  --gamma=0.3 \
  --prefix=$prefix
done
# tail -f 1129.log
