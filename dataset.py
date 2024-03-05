import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import numpy as np
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import tensorflow as tf
import copy
import pickle

# NOTE: IMAGE TRANSFORMATION ACCORDING TO LESA
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear interpolation
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear interpolation
    # transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_extract_data(dir_, json_file, type):
    assert os.path.exists(dir_), ('{} does not exist'.format(dir_))
    assert os.path.isfile(json_file), ('{} does not exist'.format(json_file))
    # assert len(cats)!=0 , ('{} should be >0'.format(len(cats)))
    return DatasetExtract(dir_, json_file, type)


class DatasetExtract(Dataset):
    def __init__(self, dir_, json_file, type):
        super(DatasetExtract, self).__init__()

        self.json_file = json_file
        self.data_ = json.load(open(json_file, 'r'))
        self.dir_ = dir_
        self.keys_ = list(self.data_.keys())
        self.type = type

    def __len__(self):
        return len(self.keys_)

    def __getitem__(self, index):
        key = self.keys_[index]#进入循环了是吧，调不出来
        label = self.data_[key]

        filename = os.path.join(self.dir_, key)
        # print(filename)
        img = Image.open(filename).convert('RGB')
        img = transform(img)
        # raw_img = Image.open(filename)
        # img = transform(raw_img)

        filename = filename.split('/')[-1]        #以‘/’为分割符，保留最后一段
        # json_file = self.json_file.split('/')[-1]
        if self.type == 'Train':
            labels = torch.tensor(np.int_(label['labels_925']))
            return filename, img, labels
        else:
            label_1006 = torch.tensor(np.int_(label['labels_1006']))
            label_81 = torch.tensor(np.int_(label['labels_81']))
            return filename, img, label_1006, label_81

        # for ONLY coco
        # key = os.path.split(key)[-1]
        # if os.path.split( self.dir_)[-1] != 'images':
        #     self.dir_ = self.dir_ + '/images'
        # print(self.dir_)


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_dir, anno_json, label_set=None, transform=None, n_val=0, mode="train",
                 return_filename=False, val_ids=None, val_cats=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_dir: image directory.
            anno_json: coco annotation file path.
            label_set: list of labels, IDs or names.
            transform: image transformation function, callable.
        """
        assert n_val >= 0

        self.coco = COCO(anno_json)
        self.image_dir = image_dir
        self.label_set = label_set
        self.return_filename = return_filename
        self.transform = transform
        self.mode = mode
        if label_set is not None:
            if not isinstance(label_set, list):
                raise ValueError(f"label_set must be a list, but got {type(label_set)}")
            if isinstance(label_set[0], str):
                self.cat_ids = sorted(self.coco.getCatIds(catNms=label_set))
            else:
                self.cat_ids = sorted(label_set)
        else:
            self.cat_ids = sorted(self.coco.getCatIds())

        self.ids = list(sorted(self.coco.imgs.keys()))
        if n_val > 0 and val_cats is not None:
            if isinstance(val_cats[0], str):
                val_cat_ids = sorted(self.coco.getCatIds(catNms=val_cats))
            else:
                val_cat_ids = val_cats

            val_ids = copy.deepcopy(self.ids)
            val_ids = self.filter_image_list(val_ids, val_cat_ids)
            self.val_ids = sorted(val_ids[:n_val])
            self.ids = [x for x in self.ids if x not in self.val_ids]
        self.ids = self.filter_image_list(self.ids, self.cat_ids)
        # if mode == "train":
        #     if n_val > 0 and val_cats is not None:
        #         if isinstance(val_cats[0], str):
        #             val_cat_ids = sorted(self.coco.getCatIds(catNms=val_cats))
        #         else:
        #             val_cat_ids = val_cats
        #
        #         val_ids = copy.deepcopy(self.ids)
        #         val_ids = self.filter_image_list(val_ids, val_cat_ids)
        #         self.val_ids = sorted(val_ids[:n_val])
        #         self.ids = [x for x in self.ids if x not in self.val_ids]
        #     self.ids = self.filter_image_list(self.ids, self.cat_ids)
        # elif mode == "val":
        #     assert val_ids is not None
        #     self.ids = val_ids
        # else:  # otherwise it is in test mode, and we use all images
        #     self.ids = self.filter_image_list(self.ids, self.cat_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Returns one data pair (image and labels)."""
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        annotation = coco.loadAnns(ann_ids)
        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        if self.mode == 'train':
            labels = torch.zeros(len(self.cat_ids))
        else:
            labels = torch.zeros(len(self.cat_ids)) -1
        for ann in annotation:
            cat = ann['category_id']
            idx = self.cat_ids.index(cat)
            labels[idx] = 1
        if self.return_filename:
            return path, image, labels
        return image, labels

    def get_img_ids(self, catIds, union=True):
        ids = set()
        for i, catId in enumerate(catIds):
            if i == 0 and len(ids) == 0:
                ids = set(self.coco.catToImgs[catId])
            else:
                if union:
                    ids |= set(self.coco.catToImgs[catId])
                else:
                    ids &= set(self.coco.catToImgs[catId])
        return list(ids)

    def get_img_ids_tight(self, catIds):
        ids = []
        coco = self.coco
        all_imgs = self.coco.imgs.keys()
        for iid in all_imgs:
            ann_ids = coco.getAnnIds(imgIds=iid, iscrowd=None)
            annotation = coco.loadAnns(ann_ids)
            flag = True
            for ann in annotation:
                cat = ann['category_id']
                if cat not in catIds:
                    flag = False
            if flag:
                ids.append(iid)
        return ids

    def filter_image_list(self, ids, cat_ids):
        """
        filter out images with no labels
        :return:
        """
        valid_ids = []
        for i in ids:
            coco = self.coco
            ann_ids = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
            annotation = coco.loadAnns(ann_ids)
            labels = np.zeros(len(cat_ids))
            for ann in annotation:
                cat = ann['category_id']
                idx = cat_ids.index(cat)
                labels[idx] = 1
            if np.sum(labels) > 0:
                valid_ids.append(i)
        return valid_ids


class OpenimagesDataset(Dataset):
    def __init__(self, image_dir, anno_json, label_set=None, transform=None,mode = 'train'):
        self.image_dir = image_dir
        self.anno_json = anno_json
        self.transform = transform
        self.label_set = label_set
        self.mode = mode
        self.images = anno_json.drop_duplicates("ImageID",ignore_index=True)
        print('len(self.images)',len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # print("index",index)
        image_id = self.images.iloc[index,0]
        # image_id = image_id+'.jpg'
        # print('image_id',image_id)
        if self.mode == 'train':
            image_train_dir = image_id[0]
            filename = self.image_dir + '/train_{}/{}.jpg'.format(image_train_dir, image_id)
        if self.mode == 'test':
            filename = self.image_dir + '/test/{}.jpg'.format(image_id)
        if self.mode == 'val':
            filename = self.image_dir + '/validation/{}.jpg'.format(image_id)
        try:
            img = Image.open(filename).convert('RGB')
            img = self.transform(img)
            df_img_label = self.anno_json.query('ImageID=="{}"'.format(image_id))
            label = np.zeros(len(self.label_set), dtype=np.int32)
            for index, row in df_img_label.iterrows():
                if row['LabelName'] in self.label_set:
                    idx = self.label_set.index(row['LabelName'])
                    # label[idx] = 2 * row['Confidence'] - 1
                    if row['Confidence'] == 1:
                        label[idx] = 1#2 * row['Confidence'] - 1
            return img, label
        except:
            print('filename',filename)



def LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path):
    seen_labelmap = [line.rstrip() for line in tf.io.gfile.GFile(seen_labelmap_path)]
    with open(unseen_labelmap_path, 'rb') as infile:
        unseen_labelmap = pickle.load(infile).tolist()
    # unseen_labelmap = pd.read_csv(unseen_labelmap_path).iloc[:,1].list()
    label_dict = {}
    for line in tf.io.gfile.GFile(dict_path):
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]
    return seen_labelmap, unseen_labelmap, label_dict
def get_extract_data_dd(dir_, json_file, type):
    assert os.path.exists(dir_), ('{} does not exist'.format(dir_))
    assert os.path.isfile(json_file), ('{} does not exist'.format(json_file))
    # assert len(cats)!=0 , ('{} should be >0'.format(len(cats)))
    return DatasetExtract_dd(dir_, json_file, type)
class DatasetExtract_dd(Dataset):
    def __init__(self, dir_, json_file, type):
        super(DatasetExtract_dd, self).__init__()

        self.json_file = json_file
        self.data_ = json.load(open(json_file, 'r'))
        self.dir_ = dir_
        self.keys_ = list(self.data_.keys())
        self.type = type
        file_tag1k = 'datasets/NUS-WIDE/NUS_WID_Tags/TagList1k.txt'
        file_tag81 = 'datasets/NUS-WIDE/ConceptsList/Concepts81.txt'
        self.nus_class_name_1006, self.nus_class_name_81, self.nus_class_name_925 = self._get_classes(file_tag1k, file_tag81)
        print(self.nus_class_name_1006)
    def __len__(self):
        return len(self.keys_)
    def _get_classes(self,file_tag1k, file_tag81):
        with open(file_tag1k, "r") as file:
            tag1k = np.array(file.read().splitlines())
        with open(file_tag81, "r") as file:
            tag81 = np.array(file.read().splitlines())
        tag1k = list(tag1k)
        tag81 = list(tag81)
        tag925 = list(set(tag1k).difference(set(tag81)))
        tag1k.extend(tag81)
        tag1006 = set(tag1k)
        return tag1006, tag81, tag925
    def __getitem__(self, index):
        key = self.keys_[index]
        label = self.data_[key]

        filename = os.path.join(self.dir_, key)
        print("filename1121_{}".format(index),filename)
        img = Image.open(filename).convert('RGB')
        img = transform(img)
        # raw_img = Image.open(filename)
        # img = transform(raw_img)

        filename = filename.split('/')[-1]
        print('filename_{}'.format(index),filename)
        # json_file = self.json_file.split('/')[-1]
        if self.type == 'Train':
            labels = torch.tensor(np.int_(label['labels_925']))
            return filename, img, labels
        else:
            label_1006 = torch.tensor(np.int_(label['labels_1006']))
            label_81 = torch.tensor(np.int_(label['labels_81']))
            label_1006 = torch.clamp(label_1006, 0, 1)
            label_81 = torch.clamp(label_81, 0, 1)
            nus_class_name_1006 = np.array(list(self.nus_class_name_1006))
            nus_class_name_81 = np.array(self.nus_class_name_81)
            c = label_1006.bool()
            nus_class_name_1006_d = nus_class_name_1006
            lab_1006 = nus_class_name_1006_d[c]
            d = label_81.bool()
            lab_81 = nus_class_name_81[d]
            print('lab_81_{}'.format(index),lab_81)
            print('lab_1006_{}'.format(index), lab_1006)
            return filename, img, label_1006, label_81

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    #
    # path = 'datasets/OpenImages/2018_04/'
    # seen_labelmap_path = path + '/classes-trainable.txt'
    # dict_path = path + '/class-descriptions.csv'
    # unseen_labelmap_path = path + 'unseen_labels.pkl'#'/top_400_unseen.csv'
    # seen_labelmap, unseen_labelmap, label_dict = LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path)
    # # images = pd.read_csv(path + 'test/images.csv')['ImageID']
    # annot = pd.read_csv(path + 'my_train/train-annotations-human.csv')
    # dataset = OpenimagesDataset('/home/ttl/dataset/openimages/images',
    #                             # images,
    #                             annot,
    #                             transform=transform, label_set=seen_labelmap,mode = 'train')
    # loader = DataLoader(dataset,
    #                     batch_size=10,
    #                     num_workers=4,
    #                     shuffle=False)
    src = 'datasets/NUS-WIDE/features/'
    train_dataset = get_extract_data_dd(
        dir_=os.path.join('/home/ttl/dataset/NUS-WIDE/nuswide/Flickr/'),
        # /root/mlc-zsl/datasets/NUS-WIDE/nuswide/Flickr/
        json_file=os.path.join(src, 'nus_wide_test.json'), type='Test')
    loader = DataLoader(dataset=train_dataset, batch_size=60, shuffle=True,
                                   num_workers=8, drop_last=False, pin_memory=False)
    tqdm_obj = tqdm(loader, ncols=80)
    for step, batch in enumerate(tqdm_obj):
        _, img, labels = batch[0], batch[1].cuda(), batch[2].cuda()
        break
