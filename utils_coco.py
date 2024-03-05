import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
num_seen = 65
word = np.loadtxt('datasets/COCO/word_glo.txt', dtype='float32', delimiter=',')
word_seen = word[:,:num_seen]
word_unseen = word[:,num_seen:]
wordname_all = open('datasets/COCO/cls_names_test_coco.csv').read().split("\n")
all_class_mapping = []
for idx in range(int(len(wordname_all)) - 1):
    all_class_mapping.append(wordname_all[idx].split(',')[0])
# wordname_lines = open('MSCOCO/cls_names_test_coco.csv').read().split("\n")
wordname_seen = open('datasets/COCO/cls_names_seen_coco.csv').read().split("\n")
print()
