import  json
from pypinyin import pinyin, Style
from tqdm import tqdm

whole_pinyin_file_path = '/home/ljh/CSC/Enhanced_Syllable_Feature/data/ablation_ft_data/train_all'
vocab_file = '/home/ljh/model/ChineseBERT-base/vocab.txt'



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

class Pinyin(object):
    """docstring for Pinyin"""
    def __init__(self):
        super(Pinyin, self).__init__()
        self.shengmu = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
        self.yunmu = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un', 'uo', 'v', 've']
        self.shengdiao= ['1', '2', '3', '4', '5']
        self.sm_size=len(self.shengmu)+1
        self.ym_size=len(self.yunmu)+1
        self.sd_size=len(self.shengdiao)+1

    def get_sm_ym_sd(self, pinyin):
        s=pinyin
        assert isinstance(s, str),'input of function get_sm_ym_sd is not string'
        if len(s) == 0:
            return None, None, None
        if s[-1] not in '12345':
            s += '5'
        assert s[-1] in '12345',f'input of function get_sm_ym_sd is not valid,{s}'
        sm, ym, sd = None, None, None
        for c in self.shengmu:
            if s.startswith(c):
                sm = c
                break
        if sm == None:
            ym = s[:-1]
        else:
            ym = s[len(sm):-1]
        sd = s[-1]
        return sm, ym, sd
    
    def get_sm_ym_sd_labels(self, pinyin):
        sm, ym, sd=self.get_sm_ym_sd(pinyin)
        return self.shengmu.index(sm)+1 if sm in self.shengmu else 0, \
            self.yunmu.index(ym)+1 if ym in self.yunmu else 0, \
                self.shengdiao.index(sd)+1 if sd in self.shengdiao else 0
pho_convertor = Pinyin()

with open(vocab_file, 'r') as f:
    all = [line.strip() for line in f.readlines() if len(line.strip()) == 1 and is_chinese_token(line.strip())]

result = pinyin(all, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
pinyin_list = ['not chinese']
for pronunciation in result:
    for py in pronunciation:
        if py != 'not chinese' and py not in pinyin_list:
            pinyin_list.append(py)

pinyin_map_path ='/home/ljh/model/ChineseBERT-base/config/pinyin_map.json'
with open(pinyin_map_path, 'r', encoding='utf8') as f:
    idx2char = json.load(f)['idx2char']

def tensor2pinyin(tensor):
    string = ''
    for id in tensor:
        if id == 0:
            return string
        else:
            string += idx2char[id]
    return string

def syllable_parser(string):
    return pho_convertor.get_sm_ym_sd(string)

def counter_whole_pinyin(file):
    with open(file, 'r', encoding='utf8') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    print('all examples:',len(data))
    total_num = 0
    equal_pinyin_num = 0
    unequal_num = 0
    equal_0_num = 0
    equal_1_num = 0
    equal_2_num = 0
    for example in tqdm(data):
        positions = []
        for index, (a,b) in enumerate(zip(example['input_ids'],example['label'])):
            if a != b:
                total_num += 1
                positions.append(index)
        for position in positions:
            input = tensor2pinyin(example['pinyin_ids'][position])
            output = pinyin_list[example['pinyin_label'][position]]
            if input != output:
                unequal_num +=1
                equal_012 = 0
                for input_syllable, output_syllable in zip(syllable_parser(input),syllable_parser(output)):
                    if input_syllable == output_syllable:
                        equal_012 += 1
                if equal_012 == 0:
                    equal_0_num += 1
                elif equal_012 == 1:
                    equal_1_num += 1
                else:
                    equal_2_num += 1
    equal_pinyin_num = total_num - unequal_num
    print(total_num, equal_pinyin_num, unequal_num, equal_0_num, equal_1_num, equal_2_num)


counter_whole_pinyin(whole_pinyin_file_path)

# 389068 73542 315526 159237 80669 75620
#0.189/0.811  0.409/0.207/0.194