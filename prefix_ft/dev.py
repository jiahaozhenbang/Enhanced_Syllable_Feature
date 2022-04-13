import argparse
import os
from unittest import result
from pytorch_lightning import Trainer
from prefix_ft.train import PrefixCSC
import re


def remove_de(input_path, output_path):
    with open(input_path) as f:
        data = f.read()

    data = re.sub(r'\d+, 地(, )?', '', data)
    data = re.sub(r'\d+, 得(, )?', '', data)
    data = re.sub(r', \n', '\n', data)
    data = re.sub(r'(\d{5})\n', r'\1, 0\n', data)

    with open(output_path, 'w') as f:
        f.write(data)

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--ckpt_path", required=True, type=str, help="ckpt file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--label_file", default='/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/dev14.lbl',
         type=str, help="label file")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    return parser

def get_file_list(data_dir):
    files = []
    for maindir, _, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            files.append(apath)
    return files

def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    cp_list = get_file_list(args.ckpt_path)

    max_df = 0
    max_df_cp = cp_list[0]

    for index, cp in enumerate(cp_list):
        args.ckpt_path = cp
        print('model assignment')
        model = PrefixCSC.load_from_checkpoint(args.ckpt_path, batch_size = args.batch_size, label_file = args.label_file)
        print('model load completed')
        trainer = Trainer.from_argparse_args(args)
        print('trainer initialization completed')
        if '14' in args.label_file:
            output=trainer.validate(model=model,dataloaders=model.dev14_dataloader(),ckpt_path=args.ckpt_path)
        elif '13'in args.label_file:
            output=trainer.validate(model=model,dataloaders=model.dev13_dataloader(),ckpt_path=args.ckpt_path)
        else:
            output=trainer.validate(model=model,dataloaders=model.dev15_dataloader(),ckpt_path=args.ckpt_path)
        # print(output[:3])
        # from metrics.metric import Metric
        # metric = Metric(vocab_path=args.bert_path)
        # pred_txt_path = os.path.join(args.save_path, "preds.txt")
        # pred_lbl_path = os.path.join(args.save_path, "labels.txt")
        # results = metric.metric(
        #         batches=output,
        #         pred_txt_path=pred_txt_path,
        #         pred_lbl_path=pred_lbl_path,
        #         label_path=args.label_file,
        #         should_remove_de=True if '13' in args.label_file else False
        #     )
        results = output[-1]
        if results['df'] >= max_df:
            max_df = results['df']
            max_df_cp = args.ckpt_path
            print(results)
        print(index)
    print(max_df_cp)
        
    


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()