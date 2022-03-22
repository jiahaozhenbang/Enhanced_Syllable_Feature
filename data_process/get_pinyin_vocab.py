



import json
from pypinyin import pinyin, Style


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

with open('/home/ljh/model/ChineseBERT-base/vocab.txt', 'r') as f:
        all = [line.strip() for line in f.readlines() if len(line.strip()) == 1 and is_chinese_token(line.strip())]
        print(len(all), all[:10])

result = pinyin(all, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
pinyin_list = ['not chinese']
for pronunciation in result:
    for py in pronunciation:
        if py != 'not chinese' and py not in pinyin_list:
            pinyin_list.append(py)
print(pinyin_list[:10], len(pinyin_list))

# pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
# pinyin_locs = {}
# # get pinyin of each location
# for index, item in enumerate(pinyin_list):