#!/usr/bin/env bash
# -*- coding: utf-8 -*-



REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


accumulate_grad_batches=1
epoch=20
lr=3e-5
for bs in 48 32 24 16
do
    OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/decoupled/lr${lr}bs${bs}
    mkdir -p $OUTPUT_DIR
    CUDA_VISIBLE_DEVICES=0  python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/self_paced_train.py \
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
    --ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/further_pretrain/322_decoupled/AM/AM.ckpt
    
    sleep 3
done
# nohup bash /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/scripts/weight_train/lr3.sh 2>&1 >/home/ljh/CSC/Enhanced_Syllable_Feature/finetune/log/weight/lr3.log &
# tail -f /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/log/weight/lr3.log