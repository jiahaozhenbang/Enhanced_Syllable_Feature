
import tokenizers
from tokenizers import BertWordPieceTokenizer
from pypinyin import pinyin, Style
import argparse
import sys
from typing import List
import opencc
import json
import random
import collections
import os
from tqdm import tqdm
sys.path.insert(0,  '/home/ljh/CSC/Enhanced_Syllable_Feature')
from datasets.utils import pho_convertor

converter = opencc.OpenCC('t2s.json')


def traditional_to_simple(text):
    text = converter.convert(text)
    text = text.replace('著', '着').replace('妳', '你')
    return text


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
    if len(token) == 1:
        return _is_chinese_char(ord(token))
    # if token.startwith('##') and len(token)==3:
    #     return  _is_chinese_char(ord(token[2]))
    return False

def get_chinese_list():
    vocab_file = '/home/ljh/model/ChineseBERT-base/vocab.txt'
    chinese_list = []
    with open(vocab_file, 'r', encoding='utf-8', errors='ignore')as f:
        for line in f.readlines():
            token = line.strip()
            if len(token) == 1 and _is_chinese_char(ord(token)):
                chinese_list.append(converter.convert(token))
    chinese_list = list(set(chinese_list))
    return chinese_list

chinese_list  = get_chinese_list()

def get_confusionset():
    with open('/home/ljh/github/CHINESE_GRC_DATA/merge_confusion.json', 'r', encoding='utf-8') as f:
        confusionset = json.load(f)
    newconfusion = {}
    for key in confusionset:
        newconfusion[key] = [val for val in confusionset[key] if val in chinese_list]
    confusionset = newconfusion
    return confusionset


confusionset = get_confusionset()


def adjust_confusionset(args):
    global confusionset
    print('adjust confusionset')
    with open(os.path.join(args.output_txt_dir, 'counter.json'), 'r') as f:
        counter = json.load(f)

    zi_list = sorted(list(counter.keys()), key= lambda x: counter[x])
    chinese_list = zi_list[int(len(zi_list) * 0.4) : ]
    for key in confusionset:
        confusionset[key] = [val for val in confusionset[key] if val in chinese_list]
    new_confusion = {}
    for key in confusionset:
        if key in chinese_list and len(confusionset[key]) > 0:
            new_confusion[key] = confusionset[key]
    print('fix len',len(new_confusion))
    confusionset = new_confusion
    import numpy as np 
    def statistics(num_list):
        print('avg min/max',sum(num_list) / len(num_list), min(num_list), max(num_list))
        sorted_list = sorted(num_list)
        for quantile in np.arange(0,1,0.1):
            print('quantile', quantile, sorted_list[int(len(sorted_list) * quantile)])
    statistics([len(val) for _, val in confusionset.items()])

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

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
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

    def convert_sentence_to_shengmu_yunmu_shengdiao_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
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


def get_file_list(data_dir):
    files = []
    for maindir, _, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            files.append(apath)
    return files


def get_txt(args):
    chinese_list = []
    with open(args.vocab_file, 'r', encoding='utf-8', errors='ignore')as f:
        for line in f.readlines():
            token = line.strip()
            if len(token) == 1 and _is_chinese_char(ord(token)):
                chinese_list.append(converter.convert(token))
    chinese_list = list(set(chinese_list))
    counter = {}
                            
    for name in ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM']:
        print(f'get {name} txt')
        input_files = get_file_list(os.path.join(args.input_data_dir, name))
        with open(os.path.join(args.output_txt_dir,name), 'w', encoding= 'utf8') as f:
            for file in input_files:
                with open(file, 'r', encoding= 'utf8') as g:
                    for line in g.readlines():
                        try:
                            _example = json.loads(line)
                            text = _example['text']
                        except:
                            continue
                        paragraphs = text.split('\n\n')[1:]
                        for paragraph in paragraphs:
                            f.write(paragraph+'\n')
                            for char in paragraph:
                                if char in chinese_list:
                                    if char in counter:
                                        counter[char] += 1
                                    else:
                                        counter[char] = 1
    with open(os.path.join(args.output_txt_dir, 'counter.json'), 'w', encoding= 'utf8') as f:
        json.dump(counter,f, ensure_ascii= False, indent= 1)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, input_ids, pinyin_ids, label, pinyin_label):
        self.input_ids, self.pinyin_ids, self.label, self.pinyin_label = input_ids, pinyin_ids, label, pinyin_label

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [x for x in self.input_ids]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(open(output_file, 'a', encoding='utf8',))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):

        features = collections.OrderedDict()
        features["input_ids"] = instance.input_ids
        features["pinyin_ids"] = instance.pinyin_ids
        features["label"] = instance.label
        features["pinyin_label"] = instance.pinyin_label

        writers[writer_index].write(json.dumps(
            features, ensure_ascii=False)+'\n')
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    for writer in writers:
        writer.close()

    print("Wrote %d total instances", total_written)


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, masked_lm_prob, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = []
    for input_file in input_files:
        print('processing '+str(input_file))
        with open(input_file, "r", encoding= 'utf8') as reader:
            for line in tqdm(reader.readlines(), desc= input_file):
                if not line:
                    continue
                else:
                    text = line.strip()
                encoded = tokenizer.encode(text)
                tokens = encoded.tokens[1:-1]
                if '[UNK]' in tokens:
                    continue
                if tokens:
                    all_documents.append([])
                    all_documents[-1].append((tokens, text, encoded))

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.get_vocab().keys())
    instances = []
    for i in range(dupe_factor):
        for document_index in tqdm(range(len(all_documents)), desc= f'get instance dupe{i}'):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, masked_lm_prob, vocab_words, rng, tokenizer))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, masked_lm_prob, vocab_words, rng, tokenizer):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    target_seq_length = max_num_tokens

    instances = []
    i = 0
    while i < len(document):
        tokens, text, encoded = document[i]
        if len(tokens) <= target_seq_length:
            (input_ids, pinyin_ids, label, pinyin_label) = create_masked_lm_predictions(
                tokens, masked_lm_prob, vocab_words, rng, text, encoded, tokenizer)
            instance = TrainingInstance(
                input_ids, pinyin_ids, label, pinyin_label)
            instances.append(instance)
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, vocab_words, rng, text, encoded, tokenizer):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(text):
        if is_chinese_token(token):
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_text = list(text)

    num_to_predict =  max(1, int(round(len(cand_indexes) * masked_lm_prob)))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        for index in index_set:
            covered_indexes.add(index)

            masked_lms.append(MaskedLmInstance(
                index=index, label=text[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    # def fix_token(token):
    #     if token.startswith('##') and len(token) == 3 and is_chinese_token(token[2]):
    #         return token[2]
    #     return token
    for p in masked_lms:
        token = output_text[p.index]
        if token in confusionset:
            if rng.random() < 0.8:
                masked_token = confusionset[token][rng.randint(
                        0, len(confusionset[token]) - 1)]#random.choice(confusionset[token])
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = token
                # 10% of the time, replace with random word
                else:
                    masked_token = chinese_list[rng.randint(
                        0, len(chinese_list) - 1)]

            # output_tokens[p.index] = fix_token(masked_token)
            output_text[p.index] = masked_token

    input_text = ''.join(output_text)
    encoded_input = tokenizer.encode(input_text)
    input_ids = encoded_input.ids
    pinyin_ids = token2pinyin.convert_sentence_to_pinyin_ids(
        input_text, encoded_input)
    label = encoded.ids
    pinyin_label = token2pinyin.convert_sentence_to_shengmu_yunmu_shengdiao_ids(
        text, encoded)

    # print([token for token in ['[CLS]']+output_tokens+['[SEP]']])
    # print([token for token in ['[CLS]']+tokens+['[SEP]']])
    # print(pinyin_ids)
    # print(pinyin_label)
    assert len(input_ids) == len(label)
    assert len(input_ids) == len(
        pinyin_ids), f"{len(input_ids)}, {len(pinyin_ids)} {input_text} --> {pinyin_ids}, {input_ids}"
    assert len(input_ids) == len(pinyin_label), f'{text} --> {pinyin_label}'

    return (input_ids, pinyin_ids, label, pinyin_label)


def get_parser():
    parser = argparse.ArgumentParser(description="get-further-pretrain-data")
    parser.add_argument("--input_data_dir", required=True, type=str)
    parser.add_argument("--output_txt_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--adjust_confusionset", action='store_true')
    parser.add_argument(
        "--vocab_file",
        default="/home/ljh/model/ChineseBERT-base/vocab.txt",
        type=str,
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="max length of datasets"
    )
    parser.add_argument(
        "--dupe_factor", default=5, type=int
    )
    parser.add_argument(
        "--random_seed", default=12345, type=int
    )
    parser.add_argument(
        "--masked_lm_prob", default=0.15, type=float
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.output_txt_dir, 'counter.json')):
        get_txt(args)

    if args.adjust_confusionset:
        adjust_confusionset(args)
    
    rng = random.Random(args.random_seed)
    tokenizer = BertWordPieceTokenizer(
        args.vocab_file, lowercase = True)

    output_files = [os.path.join(args.output_dir, name) for name in ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM']]
    for name in ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM']:
        input_file = os.path.join(args.output_txt_dir,name)
        instances = create_training_instances(
            [input_file], tokenizer, args.max_length, args.dupe_factor, args.masked_lm_prob,
            rng)
        write_instance_to_example_files(instances, output_files)
    


if __name__ == "__main__":
    main()
"""
python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/get_further_pretrain_data.py \
    --input_data_dir /home/ljh/Data/wiki_zh --output_txt_dir /home/ljh/CSC/Enhanced_Syllable_Feature/data \
    --output_dir /home/ljh/CSC/Enhanced_Syllable_Feature/data/further_pretrain_data


python /home/ljh/CSC/Enhanced_Syllable_Feature/data_process/get_further_pretrain_data.py \
    --input_data_dir /home/ljh/Data/wiki_zh --output_txt_dir /home/ljh/CSC/Enhanced_Syllable_Feature/data \
    --output_dir /home/ljh/CSC/Enhanced_Syllable_Feature/data/further_pretrain_data_v2 --adjust_confusionset
"""