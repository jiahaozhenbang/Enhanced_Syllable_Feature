#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/epoch=29-df=80.3957-cf=78.5971.ckpt
#/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/v2_acc1/epoch20lr5e-5bs32/checkpoint/epoch=19-df=78.2135-cf=72.1133.ckpt

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/predict
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

CUDA_VISIBLE_DEVICES=2  python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/pp_predict.py \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file /home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv \
  --cuda \
  --stepsize 5