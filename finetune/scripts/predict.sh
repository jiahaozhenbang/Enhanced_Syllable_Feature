#!/usr/bin/env bash
# -*- coding: utf-8 -*-


REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/v2_acc1/epoch20lr5e-5bs32/checkpoint/epoch=19-df=78.2135-cf=72.1133.ckpt
# /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/v2_acc2/epoch30lr5e-5bs64/checkpoint/epoch=28-df=80.4324-cf=74.8108.ckpt

# /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/v2_acc1/epoch20lr5e-5bs32/checkpoint/epoch=19-df=78.2135-cf=72.1133.ckpt

OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/predict
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0  python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/predict.py \
  --model_architecture ORINGIN \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file /home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv \
  --gpus=0,

  #/home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv
#   /home/ljh/CSC/Enhanced_Syllable_Feature/data/ocr_no_dev/ocr_test.lbl
# java -jar /home/ljh/github/CHINESE_GRC_DATA/NCU_NLPLab_CSC/sighan8csc_release1.0/Tool/sighan15csc.jar \
#     -i /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/predict/pinyin/labels.txt \
#     -t SIGHAN15_Toy_Truth.txt -o SIGHAN15_Toy_Evaluation.txt