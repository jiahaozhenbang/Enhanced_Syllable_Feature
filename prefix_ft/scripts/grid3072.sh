
REPO_PATH=/home/ljh/CSC/Enhanced_Syllable_Feature
BERT_PATH=/home/ljh/model/ChineseBERT-base
DATA_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


accumulate_grad_batches=1
epoch=20
lr=5e-5
bs=32
mid_dim=3072

for preseqlen in 20 10 5
do
OUTPUT_DIR=/home/ljh/CSC/Enhanced_Syllable_Feature/outputs/finetune/prefix_ft/prefixlen${preseqlen}_mid_dim${mid_dim}_epoch${epoch}
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=3 python -u /home/ljh/CSC/Enhanced_Syllable_Feature/prefix_ft/prefix_and_finetuning.py \
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
--ckpt_path /home/ljh/CSC/Enhanced_Syllable_Feature/outputs/further_pretrain/123_v2/AM/AM.ckpt
sleep 3
done