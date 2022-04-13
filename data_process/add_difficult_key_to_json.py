import json
import argparse
import os
from typing import List
import sys

from tqdm import tqdm

sys.path.insert(0,  '/home/ljh/CSC/Enhanced_Syllable_Feature')

import tokenizers
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from datasets.utils import pho_convertor

tokenizer = BertWordPieceTokenizer('/home/ljh/model/ChineseBERT-base/vocab.txt')

def convert_sentence_to_shengmu_yunmu_shengdiao_ids(sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
    # get pinyin of a sentence
    pinyin_list = pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True,heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
    pinyin_locs = {}
    # get pinyin of each location
    for index, item in enumerate(pinyin_list):
        pinyin_string = item[0]
        # not a Chinese character, pass
        if pinyin_string == "not chinese":
            continue
        pinyin_locs[index] = pho_convertor.get_sm_ym_sd_labels(pinyin_string)

    # find chinese character location, and generate pinyin ids
    pinyin_labels = []
    for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
        if offset[1] - offset[0] != 1:
            pinyin_labels.append((0,0,0))
            continue
        if offset[0] in pinyin_locs:
            pinyin_labels.append(pinyin_locs[offset[0]])
        else:
            pinyin_labels.append((0,0,0))

    return pinyin_labels

def write_data_to_txt(data, out_file):
    with open(out_file, 'w', encoding='utf8',) as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False)+'\n')
    print("Wrote %d total instances to %s", len(data), out_file)

def add_difficult_to_json(file, out_file):

    with open(file, 'r' ,encoding='utf8') as f:
        data = [json.loads(line) for line in list(f.readlines())]

    for example in tqdm(data):
        difficult_list = []
        src, tgt = example['src'], example['tgt']
        src_tokenizer_output = tokenizer.encode(src)
        tgt_tokenizer_output = tokenizer.encode(tgt)

        src_pinyin = convert_sentence_to_shengmu_yunmu_shengdiao_ids(src, src_tokenizer_output)
        tgt_pinyin = convert_sentence_to_shengmu_yunmu_shengdiao_ids(tgt, tgt_tokenizer_output)

        for index, (src_id, tgt_id) in enumerate(zip(src_tokenizer_output.ids, tgt_tokenizer_output.ids)):
            if src_id != tgt_id :
                cnt = 0
                for i in range(3):
                    if src_pinyin[index][i] != tgt_pinyin[index][i]:
                        cnt += 1
                difficult_list.append(cnt)
        example['difficult_list'] = difficult_list
    
    write_data_to_txt(data, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    # parser.add_argument('--vocab_path', default='/home/ljh/model/ChineseBERT-base/vocab.txt')
    
    args = parser.parse_args()
    add_difficult_to_json(args.data_path, args.output_path)

"""
python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/add_difficult_key_to_json.py \
    --data_path /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/train13 \
    --output_path /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/train13_with_difficult_list

python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/add_difficult_key_to_json.py \
    --data_path /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/train14 \
    --output_path /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/train14_with_difficult_list

python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/add_difficult_key_to_json.py \
    --data_path /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/train15 \
    --output_path /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/train15_with_difficult_list
"""