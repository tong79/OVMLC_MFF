#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001 ,help='initial learning rate')
parser.add_argument('--lr_min', type=float, default=0.0002 ,help='minimum lr for scheduler drop')
parser.add_argument('--train_full_lr', type=float, default=0.0002 ,help='lr for finetuning')
parser.add_argument('--workers', type=int,help='number of data loading workers', default=16)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--train', action='store_true',default=False, help='enables cuda')
parser.add_argument('--train_full_data', action='store_true',default=False, help='Only train a pretrained model')

parser.add_argument('--eval_interval', type=int, default=2)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=10000)


parser.add_argument('--save_path', type=str, default='nus_results', help='details regarding the code')
parser.add_argument('--SESSION', type=str, default='nus_wide_paper', help='MODEL NAME')
parser.add_argument('--job_id', type=str, default='14567', help='file job id')

parser.add_argument('--heads', type=int, default=4, help='Heads for region Atn')

parser.add_argument('--cosinelr_scheduler', action='store_true',default=False, help='Run with lr scheduler')
parser.add_argument('--summary', type=str, default='Summary', help='Summary of Expt')
parser.add_argument('--src', type=str,default="datasets")

parser.add_argument('--nseen_class', type=int, default=925,help='number of seen classes')
parser.add_argument('--nclass_all', type=int, default=1006,help='number of all classes')

parser.add_argument('--channel_dim', type=int, default=256,help='conv channel dim')
parser.add_argument('--num_warmup_steps', type=float, default=0.05,help='num_warmup_steps')
parser.add_argument('--model_path', type=str, default='models/',help='save model_path')
opt = parser.parse_args()
