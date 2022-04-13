import argparse

import pickle
import random
from paddle import rand

from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from pypinyin import pinyin, Style
import os
import sys
import json
import random
sys.path.insert(0,  '/home/ljh/CSC/Enhanced_Syllable_Feature')
from datasets.utils import pho_convertor


class hanzi2pinyin():

    def __init__(self, chinese_bert_path, max_length: int = 512):
        self.vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        self.config_path = os.path.join(chinese_bert_path, 'config')
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        # load pinyin map dict
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

    def convert_sentence_to_shengmu_yunmu_shengdiao_ids(self, sentence, tokenizer_output):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True,
                             heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            pinyin_locs[index] = pho_convertor.get_sm_ym_sd_labels(
                pinyin_string)

        # find chinese character location, and generate pinyin ids
        pinyin_labels = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_labels.append((0, 0, 0))
                continue
            if offset[0] in pinyin_locs:
                pinyin_labels.append(pinyin_locs[offset[0]])
            else:
                pinyin_labels.append((0, 0, 0))

        return pinyin_labels


token2pinyin = hanzi2pinyin('/home/ljh/model/ChineseBERT-base')

def the_whole_train_data_to_pickle(data_path, output_dir, vocab_path, max_len ):
    def _build_dataset(data_path):
        print('processing ', data_path)
        return build_dataset(
        data_path=data_path,
        vocab_path=vocab_path,
        max_len=max_len
    )
    sighan13_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan13-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan13-2.tsv'))
    sighan14_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan14-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan14-2.tsv'))
    sighan15_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan15-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan15-2.tsv'))

    def write_data_to_txt(data, out_file):
        with open(out_file, 'w', encoding='utf8',) as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False)+'\n')
        print("Wrote %d total instances to %s", len(data), out_file)
    
    write_data_to_txt(sighan13_trainset, os.path.join(output_dir, 'train13'))
    write_data_to_txt(sighan14_dataset, os.path.join(output_dir, 'train14'))
    write_data_to_txt(sighan15_dataset, os.path.join(output_dir, 'train15'))


def all_train_data_to_pickle(data_path, output_dir, vocab_path, max_len ):
    def _build_dataset(data_path):
        print('processing ', data_path)
        return build_dataset(
        data_path=data_path,
        vocab_path=vocab_path,
        max_len=max_len
    )
    sighan13_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan13-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan13-2.tsv'))
    sighan14_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan14-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan14-2.tsv'))
    sighan15_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan15-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan15-2.tsv'))
    wang27k_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.wang27k.tsv'))


    train_all = sighan13_trainset + sighan14_dataset + sighan15_dataset + wang27k_trainset
    random.shuffle(train_all)

    def write_data_to_txt(data, out_file):
        with open(out_file, 'w', encoding='utf8',) as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False)+'\n')
        print("Wrote %d total instances to %s", len(data), out_file)

    write_data_to_txt(train_all, os.path.join(output_dir, 'train_all_no_dev'))


def all_train_data_to_pickle_with_tgt_pinyinid(data_path, output_dir, vocab_path, max_len ):
    def _build_dataset(data_path):
        print('processing ', data_path)
        return build_dataset_with_tgt_pinyinid(
        data_path=data_path,
        vocab_path=vocab_path,
        max_len=max_len
    )
    sighan13_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan13-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan13-2.tsv'))
    sighan14_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan14-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan14-2.tsv'))
    sighan15_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan15-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan15-2.tsv'))
    wang27k_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.wang27k.tsv'))


    train_all = sighan13_trainset + sighan14_dataset + sighan15_dataset + wang27k_trainset
    random.shuffle(train_all)

    def write_data_to_txt(data, out_file):
        with open(out_file, 'w', encoding='utf8',) as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False)+'\n')
        print("Wrote %d total instances to %s", len(data), out_file)

    write_data_to_txt(train_all, os.path.join(output_dir, 'train_all_no_dev_with_tgt_pinyinid'))

def data_to_pickle(data_path, output_dir, vocab_path, max_len ):
    def _build_dataset(data_path):
        print('processing ', data_path)
        return build_dataset(
        data_path=data_path,
        vocab_path=vocab_path,
        max_len=max_len
    )
    sighan13_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan13-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan13-2.tsv'))
    sighan14_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan14-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan14-2.tsv'))
    sighan15_dataset = _build_dataset(data_path=os.path.join(data_path, 'train.sighan15-1.tsv')) + _build_dataset(
        data_path=os.path.join(data_path, 'train.sighan15-2.tsv'))
    wang27k_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.wang27k.tsv'))

    sample_index = random.sample(list(range(0, len(sighan14_dataset))), 500)
    dev14 = [sighan14_dataset[index] for index in sample_index]
    train14 = [sighan14_dataset[index] for index in range(0, len(sighan14_dataset)) if index not in sample_index]

    sample_index = random.sample(list(range(0, len(sighan15_dataset))), 500)
    dev15 = [sighan15_dataset[index] for index in sample_index]
    train15 = [sighan15_dataset[index] for index in range(0, len(sighan15_dataset)) if index not in sample_index]

    train_all = sighan13_trainset + train14 + train15 + wang27k_trainset
    random.shuffle(train_all)

    def write_data_to_txt(data, out_file):
        with open(out_file, 'w', encoding='utf8',) as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False)+'\n')
        print("Wrote %d total instances to %s", len(data), out_file)
    
    write_data_to_txt(dev14, os.path.join(output_dir, 'dev14'))
    write_data_to_txt(dev15, os.path.join(output_dir, 'dev15'))
    write_data_to_txt(train_all, os.path.join(output_dir, 'train_all'))

    dev13 = dev14 + dev15
    write_data_to_txt(dev13, os.path.join(output_dir, 'dev13'))



def build_dataset(data_path, vocab_path, max_len):
    # Load Data
    data_raw = []
    with open(data_path, encoding='utf8') as f:
        data_raw = [s.split('\t') for s in f.read().splitlines()]
    print(f'#Item: {len(data_raw)} from "{data_path}"')

    # Vocab
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = True)


    # Data Basic
    data = []
    for item_raw in tqdm(data_raw, desc='Build Dataset'):
        # Field: id, src, tgt
        item = {
            'id': item_raw[0],
            'src': item_raw[1],
            'tgt': item_raw[2],
        }
        assert len(item['src']) == len(item['tgt'])
        data.append(item)

        # Field: tokens_size
        encoded = tokenizer.encode(item['src'])
        tokens = encoded.tokens[1:-1]
        tokens_size = []
        for t in tokens:
            if t == '[UNK]':
                tokens_size.append(1)
            elif t.startswith('##'):
                tokens_size.append(len(t) - 2)
            else:
                tokens_size.append(len(t))
        item['tokens_size'] = tokens_size

        # Field: src_idx
        item['input_ids'] = encoded.ids
        item['pinyin_ids'] = token2pinyin.convert_sentence_to_pinyin_ids(item['src'], encoded)

        # Field: tgt_idx
        encoded = tokenizer.encode(item['tgt'])
        item['label'] = encoded.ids
        item['pinyin_label'] = token2pinyin.convert_sentence_to_shengmu_yunmu_shengdiao_ids(item['tgt'], encoded)
        assert len(item['input_ids']) == len(item['label'])


    # Trim
    if max_len > 0:
        n_all_items = len(data)
        data = [item for item in data if len(item['input_ids']) <= max_len]
        n_filter_items = len(data)
        n_cut = n_all_items - n_filter_items
        print(f'max_len={max_len}, {n_all_items} -> {n_filter_items} ({n_cut})')

    return data

def build_dataset_with_tgt_pinyinid(data_path, vocab_path, max_len):
    # Load Data
    data_raw = []
    with open(data_path, encoding='utf8') as f:
        data_raw = [s.split('\t') for s in f.read().splitlines()]
    print(f'#Item: {len(data_raw)} from "{data_path}"')

    # Vocab
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = True)


    # Data Basic
    data = []
    for item_raw in tqdm(data_raw, desc='Build Dataset'):
        # Field: id, src, tgt
        item = {
            'id': item_raw[0],
            'src': item_raw[1],
            'tgt': item_raw[2],
        }
        assert len(item['src']) == len(item['tgt'])
        data.append(item)

        # Field: tokens_size
        encoded = tokenizer.encode(item['src'])
        tokens = encoded.tokens[1:-1]
        tokens_size = []
        for t in tokens:
            if t == '[UNK]':
                tokens_size.append(1)
            elif t.startswith('##'):
                tokens_size.append(len(t) - 2)
            else:
                tokens_size.append(len(t))
        item['tokens_size'] = tokens_size

        # Field: src_idx
        item['input_ids'] = encoded.ids
        item['pinyin_ids'] = token2pinyin.convert_sentence_to_pinyin_ids(item['src'], encoded)

        # Field: tgt_idx
        encoded = tokenizer.encode(item['tgt'])
        item['label'] = encoded.ids
        item['tgt_pinyin_ids'] = token2pinyin.convert_sentence_to_pinyin_ids(item['tgt'], encoded)
        item['pinyin_label'] = token2pinyin.convert_sentence_to_shengmu_yunmu_shengdiao_ids(item['tgt'], encoded)
        assert len(item['input_ids']) == len(item['label'])


    # Trim
    if max_len > 0:
        n_all_items = len(data)
        data = [item for item in data if len(item['input_ids']) <= max_len]
        n_filter_items = len(data)
        n_cut = n_all_items - n_filter_items
        print(f'max_len={max_len}, {n_all_items} -> {n_filter_items} ({n_cut})')

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--vocab_path', default='/home/ljh/model/ChineseBERT-base/vocab.txt')
    
    parser.add_argument('--max_len', type=int, default= 512)
    args = parser.parse_args()

    # data_to_pickle(
    #     data_path=args.data_path,
    #     output_dir=args.output_dir,
    #     vocab_path=args.vocab_path,
    #     max_len=args.max_len,
    # )
    # the_whole_train_data_to_pickle(
    #     data_path=args.data_path,
    #     output_dir=args.output_dir,
    #     vocab_path=args.vocab_path,
    #     max_len=args.max_len,
    # )
    # all_train_data_to_pickle(
    #     data_path=args.data_path,
    #     output_dir=args.output_dir,
    #     vocab_path=args.vocab_path,
    #     max_len=args.max_len,
    # )
    all_train_data_to_pickle_with_tgt_pinyinid(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        max_len=args.max_len,
    )
"""
python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/dataset.py \
    --data_path /home/ljh/CSC/Enhanced_Syllable_Feature/data \
    --output_dir /home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data

python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/dataset.py \
    --data_path /home/ljh/CSC/Enhanced_Syllable_Feature/data \
    --output_dir /home/ljh/CSC/Enhanced_Syllable_Feature/data/ft_data_no_dev
"""
