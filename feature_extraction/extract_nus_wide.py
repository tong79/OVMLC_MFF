import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
import json
from tqdm import tqdm
import numpy as np
import h5py
from data import get_extract_data
import ssl
import clip
ssl._create_default_https_context = ssl._create_unverified_context

np.random.seed(1234)
import pickle


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

#
# model = Net()
# model = model.eval()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
model, preprocess = clip.load("ViT-B/32")
model.cuda()
# GPU = False
# if GPU:

    # device_ids = [i for i in range(torch.cuda.device_count())]
    # if torch.cuda.device_count() > 1:
    #     print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    #
    # if len(device_ids) > 1:
    #     model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # else:
    #     model = model.cuda()

src = '../datasets/NUS-WIDE/features/'
# jsons = ['81_only_full_nus_wide_train', '81_only_full_nus_wide_test']
jsons = ['nus_wide_train', 'nus_wide_test']#
#     img_names = []
#     for data_ in tqdm(data_loader):
#         filenames = data_[0]
#         for filename in filenames:
#             img_names.append(filename.replace('/', '_'))
#     dict_ = {'img_names':img_names}
#     save_dict(dict_, os.path.join(src, 'test_img_names.pkl'))

for json_ in jsons:
    # if json_ == '81_only_full_nus_wide_train':
    #     pickle_name='img_names_train_81.pkl'
    # else:
    #     pickle_name='img_names_test_81.pkl'

    if json_ == 'nus_wide_train':
        pickle_name = 'img_names_train.pkl'
    else:
        pickle_name = 'img_names_test.pkl'

    type_ = 'Flickr'
    dataset_ = get_extract_data(
        dir_=os.path.join('/home/shilida/multilabel/NUS-WIDE/', 'nuswide/{}'.format(type_)),
        json_file=os.path.join(src, json_ + '.json'))

    data_loader = DataLoader(dataset=dataset_, batch_size=256, shuffle=False, num_workers=16, drop_last=False)
    img_names = []

    fn = os.path.join(src, json_ + '_clip_feature.h5')
    print('当前是'+json_)
    with h5py.File(fn, mode='w') as h5f:
        for data_ in tqdm(data_loader):
            if json_ == 'nus_wide_train':
                filename, img, labels = data_[0], data_[1], data_[2]
            else:
                filename, img, lab_1006,lab_81 = data_[0], data_[1], data_[2], data_[3]
            # filename, img, lab, lab_81, lab_925 = data_[0], data_[1], data_[2], data_[3], data_[4]
            bs = img.size(0)
            img = img.cuda()
            with torch.no_grad():
                out = model.encode_image(img)
                # out = model(img)
            out = np.float32(out.cpu().numpy())
            if json_ == 'nus_wide_train':
                labels = np.int8(labels.numpy())
            else:
                labels_81 = np.int8(lab_81.numpy())#zsl的话是81类
                labels_1006 = np.int8(lab_1006.numpy())#gzsl的话是1006

            # import pdb;pdb.set_trace()

            for i in range(bs):
                if np.isnan(out[i].any()):
                    print(filename[i])
                    import pdb;

                    pdb.set_trace()
                img_names.append(filename[i])
                h5f.create_dataset(filename[i] + '-features', data=out[i], dtype=np.float32,
                                   compression="gzip")
                if json_ == 'nus_wide_train':
                    h5f.create_dataset(filename[i] + '-labels', data=labels[i], dtype=np.int8,
                                       compression="gzip")
                else:
                    h5f.create_dataset(filename[i] + '-labels', data=labels_1006[i], dtype=np.int8,
                                       compression="gzip")
                    h5f.create_dataset(filename[i] + '-labels_81', data=labels_81[i], dtype=np.int8,
                                       compression="gzip")
        dict_ = {'img_names': img_names}
        save_dict(dict_, os.path.join(src, pickle_name))

    h5f.close()
