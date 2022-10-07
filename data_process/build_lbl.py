import json

def build_lbl(data_path, lbl_path):
    with open(data_path) as f:
        rows = [json.loads(s) for s in f.read().splitlines()]

    data = []
    for example in rows:
        idx, src, tgt = example['id'], example['src'], example['tgt']
        errors = []
        for i, (a, b) in enumerate(zip(src, tgt), start=1):
            if a != b:
                errors.append((i, b))
        errors = str(errors)
        item = [idx]
        errors = eval(errors)
        if len(errors) > 0:
            for pos, correct in errors:
                item.append(str(pos))
                item.append(correct)
        else:
            item.append('0')
        data.append(', '.join(item))

    with open(lbl_path, 'w') as f:
        f.write('\n'.join(data))
        


if __name__ == '__main__':
    build_lbl(
        data_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/ocr_no_dev/ocr_test_with_tgt_pinyinid',
        lbl_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/ocr_no_dev/ocr_test.lbl',
    )
    # build_lbl(
    #     data_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/ablation_ft_data/dev15',
    #     lbl_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/ablation_ft_data/dev15.lbl',
    # )
    # build_lbl(
    #     data_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetun_data/dev14',
    #     lbl_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetun_data/dev14.lbl',
    # )
    # build_lbl(
    #     data_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetun_data/dev13',
    #     lbl_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetun_data/dev13.lbl',
    # )

