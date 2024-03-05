#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:56:19 2020

@author: akshitac8
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import model as model
import util_nus as util
from config import opt
import numpy as np
import random
import time
import os
import socket
from torch.utils.data import DataLoader
import h5py
import pickle
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import clip


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def get_classes(file_tag1k, file_tag81):
    with open(file_tag1k, "r") as file:
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81, "r") as file:
        tag81 = np.array(file.read().splitlines())
    tag1k = list(tag1k)
    tag81 = list(tag81)
    tag1k.extend(tag81)
    tag1006 = set(tag1k)
    return tag1006, tag81


print(opt)
#############################################
# setting up seeds
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
torch.set_default_tensor_type('torch.FloatTensor')
cudnn.benchmark = True  # For speed i.e, cudnn autotuner
########################################################
data = util.DATA_LOADER(opt)  ### INTIAL DATALOADER ###
model_test, preprocess = clip.load("ViT-B/32", device='cuda')
print(model_test)
src = opt.src
test_loc = os.path.join(src, 'NUS-WIDE', 'features', 'nus_wide_test_clip_feature.h5')
test_features = h5py.File(test_loc, 'r')
# for key in test_features.keys():
#     print(test_features[key].name)
#     print(test_features[key].shape)
# test_feature_keys = list(test_features.keys())#[:1000]
image_filenames = util.load_dict(os.path.join(src, 'NUS-WIDE/features', 'img_names_test.pkl'))
test_image_filenames = image_filenames['img_names']#[:100]
ntest = len(test_image_filenames)
test_batch_size = opt.test_batch_size
print(ntest)
prediction_81 = torch.empty(ntest, 81)
prediction_1006 = torch.empty(ntest, 1006)
file_tag1k = '/home/shilida/multilabel/BiAM/datasets/NUS-WIDE/NUS_WID_Tags/TagList1k.txt'
file_tag81 = '/home/shilida/multilabel/BiAM/datasets/NUS-WIDE/ConceptsList/Concepts81.txt'
lab_81 = torch.empty(ntest, 81)
lab_1006 = torch.empty(ntest, 1006)
nus_class_name_1006, nus_class_name_81 = get_classes(file_tag1k, file_tag81)
# Prompt Ensembling
nus_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    # 'a photo of a {}.',
    # 'a blurry photo of a {}.',
    # 'a black and white photo of a {}.',
    # 'a low contrast photo of a {}.',
    # 'a high contrast photo of a {}.',
    # 'a bad photo of a {}.',
    # 'a good photo of a {}.',
    # 'a photo of a small {}.',
    # 'a photo of a big {}.',
    # 'a photo of the {}.',
    # 'a blurry photo of the {}.',
    # 'a black and white photo of the {}.',
    # 'a low contrast photo of the {}.',
    # 'a high contrast photo of the {}.',
    # 'a bad photo of the {}.',
    # 'a good photo of the {}.',
    # 'a photo of the small {}.',
    # 'a photo of the big {}.',
]
for m in tqdm(range(0, ntest, test_batch_size)):
    strt = m
    endt = min(m + test_batch_size, ntest)
    bs = endt - strt
    c = m
    c += bs
    features, labels_1006, labels_81 = np.empty((bs, 512)), np.empty((bs, 1006)), np.empty((bs, 81))
    for i, key in enumerate(test_image_filenames[strt:endt]):
        # a = test_features.get('/' + key + '-features')
        features[i, :] = np.float32(test_features.get('/' + key + '-features'))
        labels_1006[i, :] = np.int32(test_features.get('/' + key + '-labels'))
        labels_81[i, :] = np.int32(test_features.get('/' + key + '-labels_81'))
    features = torch.from_numpy(features).float().cuda()
    labels_1006 = torch.from_numpy(labels_1006).long().cuda()
    labels_81 = torch.from_numpy(labels_81).long().cuda()
    with torch.no_grad():
        logits_81_zeroshot_weights = zeroshot_classifier(nus_class_name_81, nus_templates, model_test)
        logits_81 = 100.*features.float() @ logits_81_zeroshot_weights.float().cuda()
        logits_1006_zeroshot_weights = zeroshot_classifier(nus_class_name_1006, nus_templates, model_test)
        logits_1006 = 100.* features.float() @ logits_1006_zeroshot_weights.float().cuda()
    prediction_81[strt:endt, :] = logits_81
    prediction_1006[strt:endt, :] = logits_1006
    lab_81[strt:endt, :] = labels_81
    lab_1006[strt:endt, :] = labels_1006

print(("completed calculating predictions over all {} images".format(c)))
logits_81_5 = prediction_81.clone()
ap_81 = util.compute_AP(prediction_81.cuda(), lab_81.cuda())
F1_3_81, P_3_81, R_3_81 = util.compute_F1(prediction_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
F1_5_81, P_5_81, R_5_81 = util.compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)
print('ZSL AP', torch.mean(ap_81).item())
print('k=3', torch.mean(F1_3_81).item(), torch.mean(P_3_81).item(), torch.mean(R_3_81).item())
print('k=5', torch.mean(F1_5_81).item(), torch.mean(P_5_81).item(), torch.mean(R_5_81).item())
logits_1006_5 = prediction_1006.clone()
ap_1006 = util.compute_AP(prediction_1006.cuda(), lab_1006.cuda())
F1_3_1006, P_3_1006, R_3_1006 = util.compute_F1(prediction_1006.cuda(), lab_1006.cuda(), 'overall', k_val=3)
F1_5_1006, P_5_1006, R_5_1006 = util.compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

print('GZSL AP', torch.mean(ap_1006).item())
print('g_k=3', torch.mean(F1_3_1006).item(), torch.mean(P_3_1006).item(), torch.mean(R_3_1006).item())
print('g_k=5', torch.mean(F1_5_1006).item(), torch.mean(P_5_1006).item(), torch.mean(R_5_1006).item())

#
# ZSL AP 0.23456791043281555
# k=3 0.34679335355758667 0.2433333396911621 0.6033057570457458
# k=5 0.26731076836586 0.16599999368190765 0.6859503984451294
# GZSL AP 0.16202782094478607
# g_k=3 nan 0.0 0.0
# g_k=5 nan 0.0 0.0
