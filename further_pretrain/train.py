#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : ChnSetiCorp_trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/6/30 15:35
@version: 1.0
@desc  : code for ChnSetiCorp task
"""
import argparse
from genericpath import exists
import json
import os
import random
from functools import partial
from sys import prefix

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from datasets.bert_csc_dataset import PreFinetuneDataset, CSCDataset
from datasets.collate_functions import collate_to_max_length
from models.modeling_glycebert import GlyceBertForMaskedLM
from models.modeling_multitask import GlyceBertForMultiTask
from datasets.collate_functions import collate_to_max_length_with_id
from utils.random_seed import set_random_seed

set_random_seed(2333)

from typing import Any, Callable, Optional
import torchmetrics


class sentenceAccuracy:
    def __call__(self, predicts, labels) -> Any:
        acc = (predicts == labels).all(dim=1).sum() / predicts.shape[0]
        return acc


class PreFinetuneTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, prefix):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        if type(args) == dict:
            args = argparse.Namespace(**args)
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        print(args)
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(
            self.bert_dir, output_hidden_states=False
        )
        self.model = GlyceBertForMultiTask.from_pretrained(self.bert_dir)
        self.vocab_size = self.bert_config.vocab_size

        self.loss_fct = CrossEntropyLoss()
        self.acc = sentenceAccuracy()
        gpus_string = (
            self.args.gpus if not self.args.gpus.endswith(",") else self.args.gpus[:-1]
        )
        self.num_gpus = len(gpus_string.split(","))
        self.prefix = prefix

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
        )
        t_total = (
            len(self.train_dataloader())
            // self.args.accumulate_grad_batches
            * self.args.max_epochs
        )
        print("total_steps", t_total)
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids, labels=None, pinyin_labels=None):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            labels=labels,
            pinyin_labels=pinyin_labels,
            gamma=self.args.gamma if "gamma" in self.args else 0,
        )

    def compute_loss_and_acc(self, batch):
        input_ids, pinyin_ids, labels, pinyin_labels, _, _, _ = batch
        loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        outputs = self.forward(
            input_ids, pinyin_ids, labels=labels, pinyin_labels=pinyin_labels
        )
        loss = outputs.loss
        logits = outputs.logits
        # compute acc
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.acc(predict_labels * loss_mask, labels * loss_mask)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        return {"loss": loss, "log": tf_board_logs}

    # def train_epoch_end(self, outputs):
    #     """"""
    #     self.save(self.model.state_dict(), os.path.join(self.args.save_path, self.prefix))
    def validation_step(self, batch, batch_idx):
        """"""
        input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tokens_size = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        logits = self.forward(
            input_ids,
            pinyin_ids,
        ).logits
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask
        return {
            "tgt_idx": labels.cpu(),
            "pred_idx": predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }

    def validation_epoch_end(self, outputs):
        if len(outputs) == 2:
            self.log("df", 0)
            self.log("cf", 0)
            return {"df": 0, "cf": 0}
        else:
            self.log("df", 0.5)
            self.log("cf", 0.5)
            return {"df": 0.5, "cf": 0.5}

    def train_dataloader(self):
        return self.get_dataloader(prefix=self.prefix)

    def get_dataloader(self, prefix=None) -> DataLoader:
        """get training dataloader"""

        dataset = PreFinetuneDataset(
            data_path=os.path.join(self.args.data_dir, prefix),
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader
    

    def val_dataloader(self):
        dataset = CSCDataset(
            data_path='/home/ljh/CSC/Enhanced_Syllable_Feature/data/finetune_data/dev15',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = dataset.tokenizer

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle= False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader



def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument(
        "--label_file",
        default="/home/ljh/CSC/Enhanced_Syllable_Feature/data/test.sighan15.lbl.tsv",
        type=str,
        help="label file",
    )
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "--workers", type=int, default=0, help="num workers for dataloader"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="load datasets to memory to accelerate.",
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="max length of datasets"
    )
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument(
        "--save_topk", default=30, type=int, help="save topk checkpoint"
    )
    parser.add_argument("--mode", default="train", type=str, help="train or evaluate")
    parser.add_argument(
        "--warmup_proporation", default=0.01, type=float, help="warmup proporation"
    )
    parser.add_argument("--gamma", default=1, type=float, help="phonetic loss weight")
    parser.add_argument(
        "--prefix", required=True, type=str, help="phonetic loss weight"
    )
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    prefix = args.prefix
    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, prefix)):
        os.mkdir(os.path.join(args.save_path, prefix))
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.save_path, prefix), name="log"
    )

    # save args
    with open(
        os.path.join(os.path.join(args.save_path, prefix), "args.json"), "w"
    ) as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainset_list = [
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
        "AG",
        "AH",
        "AI",
        "AJ",
        "AK",
        "AL",
        "AM",
    ]

    i = trainset_list.index(prefix)

    # model = PreFinetuneTask(args, prefix)
    if i > 0:
        model = PreFinetuneTask.load_from_checkpoint(
            os.path.join(
                args.save_path, trainset_list[i - 1], trainset_list[i - 1] + ".ckpt"
            ),
            prefix=prefix,
        )
    else:
        model = PreFinetuneTask(args, prefix)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, prefix),
        filename=prefix,
        save_top_k=args.save_topk,
        monitor="cf",
        mode="max",
    )
    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )
    trainer.fit(model)
    print(prefix, "completed")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
