#!/usr/bin/env bash
# -*- coding: utf-8 -*-



REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


accumulate_grad_batches=1
epoch=50
lr=5e-5
bs=32
preseqlen=50
mid_dim=768

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/prefix/prefixlen${preseqlen}_middim${mid_dim}_epoch${epoch}
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=2 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/prefix_ft/train.py \
--preseqlen $preseqlen \
--mid_dim $mid_dim \
--bert_path $BERT_PATH \
--data_dir $DATA_DIR \
--save_path $OUTPUT_DIR \
--max_epoch=$epoch \
--lr=$lr \
--warmup_proporation 0.1 \
--batch_size=$bs \
--gamma=1 \
--gpus=1 \
--accumulate_grad_batches=$accumulate_grad_batches  \
--reload_dataloaders_every_n_epochs 1 \
--ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/v2_acc1/epoch20lr5e-5bs32/checkpoint/epoch=19-df=78.2135-cf=72.1133.ckpt \
--limit_train_batches 0.1

# nohup bash /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/scripts/weight_train/lr5.sh 2>&1 >/home/ljh/CSC/Enhanced_Syllable_Feature/finetune/log/weight/lr5.log &
# tail -f /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/log/weight/lr5.log