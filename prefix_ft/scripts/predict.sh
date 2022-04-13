#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/prefix/prefixlen20_epoch20/checkpoint/prefixlen=20.0mid_dim=512.0epoch=13-df=79.8923-cf=77.5583.ckpt

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/predict
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0  python -u /home/ljh/CSC/Enhanced_Syllable_Feature/prefix_ft/predict.py \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file /home/ljh/github/ReaLiSe-master/data/test.sighan14.lbl.tsv \
  --gpus=0,