#!/usr/bin/env bash
# -*- coding: utf-8 -*-



REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


accumulate_grad_batches=1
lr=5e-5
epoch=20
bs=16
OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/attention/lr${lr}bs${bs}_v3
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=5 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/attention_train.py \
--bert_path $BERT_PATH \
--data_dir $DATA_DIR \
--save_path $OUTPUT_DIR \
--max_epoch=$epoch \
--lr=$lr \
--warmup_proporation 0.1 \
--batch_size=$bs \
--gamma=1 \
--gpus=0, \
--accumulate_grad_batches=$accumulate_grad_batches  \
--reload_dataloaders_every_n_epochs 1 \
--ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/further_pretrain/123_v2/AM/AM.ckpt

bs=32
OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/attention/lr${lr}bs${bs}_v3
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=5 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/attention_train.py \
--bert_path $BERT_PATH \
--data_dir $DATA_DIR \
--save_path $OUTPUT_DIR \
--max_epoch=$epoch \
--lr=$lr \
--warmup_proporation 0.1 \
--batch_size=$bs \
--gamma=1 \
--gpus=0, \
--accumulate_grad_batches=$accumulate_grad_batches  \
--reload_dataloaders_every_n_epochs 1 \
--ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/further_pretrain/123_v2/AM/AM.ckpt