import argparse
import os
import re
from functools import partial
from transformers import BertConfig
from models.modeling_multitask import GlyceBertForMultiTask
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from datasets.bert_csc_dataset import CSCDataset,TestCSCDataset
from tqdm import tqdm


SMALL_CONST = 1e-15

def remove_de(input_path, output_path):
    with open(input_path) as f:
        data = f.read()

    data = re.sub(r'\d+, 地(, )?', '', data)
    data = re.sub(r'\d+, 得(, )?', '', data)
    data = re.sub(r', \n', '\n', data)
    data = re.sub(r'(\d{5})\n', r'\1, 0\n', data)

    with open(output_path, 'w') as f:
        f.write(data)

def test15_dataloader(args):
    dataset = TestCSCDataset(
        data_path='/home/ljh/github/ReaLiSe-master/data/test.sighan15.pkl',
        chinese_bert_path=args.bert_path,
        max_length=args.max_length,
    )
    from datasets.collate_functions import collate_to_max_length_with_id

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
        drop_last=False,
    )
    return dataloader

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def pp_predict(model,
        batch,
        device="cuda",
        discrim=None,
        stepsize=0.02,
        temperature=1.0,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
):
    input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tokens_size = batch
    loss_mask = (input_ids != 0)*(input_ids != 101)*(input_ids != 102).long()
    mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
    batch_size, length = input_ids.shape
    pinyin_ids = pinyin_ids.view(batch_size, length, 8)
    sequence_feature = model.bert.forward(input_ids=input_ids, pinyin_ids=pinyin_ids).last_hidden_state

    accumulate_grad = to_var(torch.zeros(sequence_feature.shape), requires_grad= True, )
    perturb_feature = torch.add(sequence_feature, accumulate_grad)

    prediction_scores, sm_scores,ym_scores,sd_scores = model.cls(perturb_feature)
    loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
    sm_mask = (torch.argmax(sm_scores, dim=-1) != pinyin_labels[...,0])
    ym_mask = (torch.argmax(ym_scores, dim=-1) != pinyin_labels[...,1])
    sd_mask = (torch.argmax(sd_scores, dim=-1) != pinyin_labels[...,2])
    phonetic_mask = sm_mask | ym_mask | sd_mask
    active_loss = (loss_mask * phonetic_mask).view(-1) == 1
    active_labels = torch.where(
        active_loss, pinyin_labels[...,0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
    )
    sm_loss = loss_fct(sm_scores.view(-1, model.cls.Phonetic_relationship.pinyin.sm_size), active_labels)
    active_labels = torch.where(
        active_loss, pinyin_labels[...,1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
    )
    ym_loss = loss_fct(ym_scores.view(-1, model.cls.Phonetic_relationship.pinyin.ym_size), active_labels)
    active_labels = torch.where(
        active_loss, pinyin_labels[...,2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
    )
    sd_loss = loss_fct(sd_scores.view(-1, model.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
    phonetic_loss=(sm_loss+ym_loss+sd_loss)/3


    phonetic_loss.backward()

    grad_norms = torch.norm(accumulate_grad.grad ) + SMALL_CONST
    grad = -stepsize * (accumulate_grad.grad / grad_norms ** gamma).data#.cpu().numpy()

    perturb_feature = torch.add(grad, sequence_feature)
    prediction_scores, sm_scores,ym_scores,sd_scores = model.cls(perturb_feature)

    predict_scores = F.softmax(prediction_scores, dim=-1)
    predict_labels = torch.argmax(predict_scores, dim=-1) * mask
    
      
    pre_predict_labels = predict_labels
    for _ in range(1):
        record_index = []
        for i,(a,b) in enumerate(zip(list(input_ids[0,1:-1]),list(predict_labels[0,1:-1]))):
            if a!=b:
                record_index.append(i)
        
        input_ids[0,1:-1] = predict_labels[0,1:-1]
        logits = model.forward(input_ids=input_ids, pinyin_ids=pinyin_ids).logits
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask

        for i,(a,b) in enumerate(zip(list(input_ids[0,1:-1]),list(predict_labels[0,1:-1]))):
            if a!=b and any([abs(i-x)<=1 for x in record_index]):
                print(ids,srcs)
                print(i+1,)
            else:
                predict_labels[0,i+1] = input_ids[0,i+1]
        if predict_labels[0,i+1] == input_ids[0,i+1]:
            break
    return {
        "tgt_idx": labels.cpu(),
        "post_pred_idx": predict_labels.cpu(),
        "pred_idx": pre_predict_labels.cpu(),
        "id": ids,
        "src": srcs,
        "tokens_size": tokens_size,
    }


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--ckpt_path", required=True, type=str, help="ckpt file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--label_file", default='/home/ljh/github/ReaLiSe-master/data/test.sighan15.lbl.tsv',
         type=str, help="label file")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--cuda", action="store_true",)
    parser.add_argument("--stepsize", type=float, default=0.02)
    return parser


def main():
    """main"""
    parser = get_parser()
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    bert_dir = args.bert_path
    bert_config = BertConfig.from_pretrained(
        bert_dir, output_hidden_states=True
    )
    model = GlyceBertForMultiTask.from_pretrained(bert_dir, output_hidden_states=True)
    print("loading from ", args.ckpt_path)
    ckpt = torch.load(args.ckpt_path, map_location=device)["state_dict"]
    new_ckpt = {}
    for key in ckpt.keys():
        new_ckpt[key[6:]] = ckpt[key]
    model.load_state_dict(new_ckpt,strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    outputs = []
    for batch in tqdm(test15_dataloader(args)):
        output = pp_predict(model, batch, device=device, stepsize= args.stepsize)
        outputs.append(output)

    from metrics.metric import Metric
    metric = Metric(vocab_path=args.bert_path)
    pred_txt_path = os.path.join(args.save_path, "preds.txt")
    pred_lbl_path = os.path.join(args.save_path, "labels.txt")
    results = metric.metric(
            batches=outputs,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=args.label_file,
            should_remove_de=True if '13'in args.label_file else False
        )
    print(results)
    for ex in outputs:
        ex['pred_idx'] = ex['post_pred_idx']
    results = metric.metric(
            batches=outputs,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=args.label_file,
            should_remove_de=True if '13'in args.label_file else False
        )
    print(results)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()