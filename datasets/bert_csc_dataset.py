#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : bert_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/7/5 11:25
@version: 1.0
@desc  : BERT model
"""
import json
import os
from typing import List

import sys

import tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
from pypinyin import pinyin, Style
from transformers import BertTokenizer
from datasets.chinese_bert_dataset import ChineseBertDataset

from datasets.utils import pho_convertor
import pickle

class CSCDataset(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        file = self.data_path
        print('processing ',file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())
        self.data = [line for line in self.data if len(json.loads(line)['input_ids']) < 192]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = json.loads(self.data[idx])
        input_ids, pinyin_ids, label,pinyin_label = example['input_ids'], example['pinyin_ids'], example['label'], example['pinyin_label']
        example_id,src,tokens_size = example['id'], example['src'], example['tokens_size']
        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_ids = torch.LongTensor(pinyin_ids).view(-1)

        label = torch.LongTensor(label)
        pinyin_label=torch.LongTensor(pinyin_label)
        return input_ids, pinyin_ids, label,pinyin_label,example_id,src,tokens_size

class Dynaimic_CSCDataset(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        file = self.data_path
        print('processing ',file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())
        self.data = [line for line in self.data if len(json.loads(line)['input_ids']) < 192]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = json.loads(self.data[idx])
        input_ids, pinyin_ids, label,pinyin_label = example['input_ids'], example['pinyin_ids'], example['label'], example['pinyin_label']
        tgt_pinyin_ids = example['tgt_pinyin_ids']
        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_ids = torch.LongTensor(pinyin_ids).view(-1)

        label = torch.LongTensor(label)
        pinyin_label=torch.LongTensor(pinyin_label)
        tgt_pinyin_ids = torch.LongTensor(tgt_pinyin_ids).view(-1)
        return input_ids, pinyin_ids, label, tgt_pinyin_ids, pinyin_label


class Dynaimic_CSCDataset_basedonLevenshtein(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        file = self.data_path
        print('processing ',file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())
        self.data = [line for line in self.data if len(json.loads(line)['input_ids']) < 192]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = json.loads(self.data[idx])
        input_ids, pinyin_ids, label,pinyin_label = example['input_ids'], example['pinyin_ids'], example['label'], example['pinyin_label']
        pron_similarity = example['pron_similarity']
        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_ids = torch.LongTensor(pinyin_ids).view(-1)

        label = torch.LongTensor(label)
        pinyin_label=torch.LongTensor(pinyin_label)
        pron_similarity = torch.LongTensor(pron_similarity)
        return input_ids, pinyin_ids, label, pron_similarity, pinyin_label

class TestCSCDataset(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert sentence to ids
        sentence=self.data[idx]['src']
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        # label = torch.LongTensor(self.data[idx]['tgt_idx'])
        sentence=self.data[idx]['tgt']
        tokenizer_output = self.tokenizer.encode(sentence)

        assert len(bert_tokens) == len(tokenizer_output.ids)
        label = torch.LongTensor(tokenizer_output.ids)
        pinyin_label=torch.LongTensor(self.convert_sentence_to_shengmu_yunmu_shengdiao_ids(sentence, tokenizer_output))
        example_id=self.data[idx]['id']
        src=self.data[idx]['src']
        tokens_size=self.data[idx]['tokens_size']
        return input_ids, pinyin_ids, label,pinyin_label,example_id,src,tokens_size

class PPLMTestCSCDataset(ChineseBertDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert sentence to ids
        sentence=self.data[idx]['src']
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        pinyin_label=torch.LongTensor(self.convert_sentence_to_shengmu_yunmu_shengdiao_ids(sentence, tokenizer_output))
        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        # label = torch.LongTensor(self.data[idx]['tgt_idx'])
        sentence=self.data[idx]['tgt']
        tokenizer_output = self.tokenizer.encode(sentence)

        assert len(bert_tokens) == len(tokenizer_output.ids)
        label = torch.LongTensor(tokenizer_output.ids)
        
        example_id=self.data[idx]['id']
        src=self.data[idx]['src']
        tokens_size=self.data[idx]['tokens_size']
        return input_ids, pinyin_ids, label,pinyin_label,example_id,src,tokens_size

def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False

def is_chinese_token(token):
    if len(token)==1:
        return _is_chinese_char(ord(token))
    # if token.startwith('##') and len(token)==3:
    #     return  _is_chinese_char(ord(token[2]))
    return False

class PreFinetuneDataset(ChineseBertDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        file = self.data_path
        print('processing ',file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = json.loads(self.data[idx])
        input_ids, pinyin_ids, label,pinyin_label = example['input_ids'], example['pinyin_ids'], example['label'], example['pinyin_label']
        # convert list to tensor
        for i in range(len(input_ids)):
            if type(input_ids[i]) != type(1):
                input_ids[i] = 103
        # try:
        #     input_ids = torch.LongTensor(input_ids)
        # except:
        #     print(input_ids)
        #     print(type(input_ids))
        #     print(self.data[idx])
        input_ids = torch.LongTensor(input_ids)
        pinyin_ids = torch.LongTensor(pinyin_ids).view(-1)
        label = torch.LongTensor(label)
        pinyin_label=torch.LongTensor(pinyin_label)
        return input_ids, pinyin_ids, label,pinyin_label,None,None,None

def get_pickle_data(file, w_file):
    print('processing ',file)
    data=[]
    with open(file, 'r' ,encoding='utf8') as f:
        line = f.readline()
        while(line):
            example = json.loads(line)
            data.append(example)
            line = f.readline()
    pickle.dump(data, open(w_file, 'wb'))
    print(data[:3])

if __name__=='__main__':
    # a=CSCDataset(data_path='/home/ljh/github/ChineseBert-main/data/test.pkl',chinese_bert_path='/home/ljh/model/ChineseBERT-base')
    # print(a[0])
    # print(pinyin(["各", "传", "统", "[MASK]", "[MASK]", "复", "音", "唱", "法", "大", "多", "是", "[MASK]", "[MASK]", "[MASK]", "[MASK]", "织", "体", "，", "罕", "见", "「", "泛", "音", "和", "声", "合", "唱", "」", "，", "此", "堪", "称", "为", "世", "界", "音", "乐", "之", "瑰", "宝", "。"], style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x]))
    # get_pickle_data('/home/ljh/CSC/joint-training/Semantic_word_mask/pretrain_data','/home/ljh/CSC/joint-training/Semantic_word_mask/data/demo/demo')
    # for name in ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM']:
    #     get_pickle_data('/home/ljh/CSC/joint-training/Semantic_word_mask/'+name,'/home/ljh/CSC/joint-training/Semantic_word_mask/data/'+name)
    a=PreFinetuneDataset(data_path='/home/ljh/CSC/joint-training/pre-finetue/data/AA',chinese_bert_path='/home/ljh/model/ChineseBERT-base')

    for i in range(a.__len__()):
        a.__getitem__(i)