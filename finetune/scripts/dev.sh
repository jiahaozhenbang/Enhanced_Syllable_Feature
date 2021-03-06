#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/decoupled/lr5e-5bs16/checkpoint

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/dev
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

CUDA_VISIBLE_DEVICES=5  python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/dev.py \
  --model_architecture ORINGIN \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/dev14.lbl \
  --gpus=0,