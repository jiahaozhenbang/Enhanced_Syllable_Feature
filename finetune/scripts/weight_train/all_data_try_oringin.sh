#!/usr/bin/env bash
# -*- coding: utf-8 -*-



REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


accumulate_grad_batches=1
epoch=6
lr=5e-5
bs=32

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/decoupled/lr${lr}bs${bs}_oringin_epoch${epoch}
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=5 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/self_paced_train_try15.py \
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
--ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/v2_acc1/epoch20lr5e-5bs32/checkpoint/epoch=19-df=78.2135-cf=72.1133.ckpt

# nohup bash /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/scripts/weight_train/lr5.sh 2>&1 >/home/ljh/CSC/Enhanced_Syllable_Feature/finetune/log/weight/lr5.log &
# tail -f /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/log/weight/lr5.log