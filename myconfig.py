#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.00005,help='initial learning rate')
parser.add_argument('--workers', type=int,help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, help='manual seed', default=42)
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--val_batch_size', type=int, default=64)
# parser.add_argument('--nseen_class', type=int, default=925,help='number of seen classes')
# parser.add_argument('--nclass_all', type=int, default=1006,help='number of all classes')
# parser.add_argument('--channel_dim', type=int, default=256,help='conv channel dim')
parser.add_argument('--num_warmup_steps', type=float, default=0.05,help='num_warmup_steps')
parser.add_argument('--save_path', type=str, default='models/',help='save model_path')
parser.add_argument('--gpu_id', type=int, default=0,help='save model_path')
parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',help='model type')
parser.add_argument('--decoder_num', type=int, default=3,help='decoder_num')
parser.add_argument('--multi_head', type=int, default=8,help='multi_head')
opt = parser.parse_args()