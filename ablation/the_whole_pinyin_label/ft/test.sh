#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/ablation_ft_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/Ablation_finetune/epoch20lr5e-5bs32/checkpoint/epoch=17-df=79.2536-cf=75.9605.ckpt

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/Ablation_finetune/predict
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

CUDA_VISIBLE_DEVICES=7  python -u /home/ljh/CSC/Enhanced_Syllable_Feature/ablation/the_whole_pinyin_label/ft/predict.py \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file /home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv \
  --gpus=0,