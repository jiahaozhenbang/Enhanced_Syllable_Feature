#!/usr/bin/env bash
# -*- coding: utf-8 -*-



REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


accumulate_grad_batches=1
epoch=20
for lr in 5e-5 2e-5
do
    for bs in 64 32
    do
        OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/gamma0.3/lr${lr}bs${bs}
        mkdir -p $OUTPUT_DIR
        CUDA_VISIBLE_DEVICES=5 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/train.py \
        --bert_path $BERT_PATH \
        --data_dir $DATA_DIR \
        --save_path $OUTPUT_DIR \
        --max_epoch=$epoch \
        --lr=$lr \
        --warmup_proporation 0.1 \
        --batch_size=$bs \
        --gpus=0, \
        --gamma 0.3 \
        --accumulate_grad_batches=$accumulate_grad_batches  \
        --reload_dataloaders_every_n_epochs 1 \
        --ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/further_pretrain/318_gamma0.3/AM/AM.ckpt
        sleep 1
    done
done