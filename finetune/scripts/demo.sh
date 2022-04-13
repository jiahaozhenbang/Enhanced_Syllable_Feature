
REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data


TIME=demo_no192
OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/${TIME}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
# CUDA_VISIBLE_DEVICES=3  python /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/train.py \
#   --bert_path $BERT_PATH \
#   --data_dir $DATA_DIR \
#   --save_path $OUTPUT_DIR \
#   --max_epoch=30 \
#   --lr=5e-5 \
#   --warmup_proporation 0.1 \
#   --batch_size=64 \
#   --gpus=0, \
#   --gamma=1 \
#   --accumulate_grad_batches 2 \
#   --reload_dataloaders_every_n_epochs 1 \
#   --limit_train_batches 0.001 
CUDA_VISIBLE_DEVICES=3 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/finetune/train.py \
  --bert_path $BERT_PATH \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --max_epoch=20 \
  --lr=5e-5 \
  --warmup_proporation 0.1 \
  --batch_size=32 \
  --gpus=0, \
  --gamma=1 \
  --accumulate_grad_batches 1 \
  --reload_dataloaders_every_n_epochs 1 \
--ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/further_pretrain/123_v2/AM/AM.ckpt
