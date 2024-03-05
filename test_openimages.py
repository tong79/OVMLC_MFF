import os
from myconfig import opt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import mymodel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import model as model
import util_openimages as util

import numpy as np
import random
import pickle
from copy import deepcopy
from tqdm import tqdm, trange
from sklearn.preprocessing import normalize
import dataset
from torch.utils.data import DataLoader
# from warmup_scheduler import GradualWarmupScheduler
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup)
from torch.cuda.amp import autocast, GradScaler
import wandb
import pandas as pd

wandb.init(project='mlc_zsl', entity='shilida')
from torch.optim import lr_scheduler


class Test:
    def __init__(self,
                 model, dataloader, device, config, labelset):
        self.model = model
        self.device = device
        self.config = config
        self.dataloder = dataloader
        att_path = os.path.join('datasets/OpenImages/wiki_contexts/',
                                'OpenImage_w2v_context_window_10_glove-wiki-gigaword-300.pkl')
        path_top_unseen = os.path.join('datasets/OpenImages/2018_04/', 'top_400_unseen.csv')
        df_top_unseen = pd.read_csv(path_top_unseen, header=None)
        self.idx_top_unseen = df_top_unseen.values[:, 0]
        assert len(self.idx_top_unseen) == 400
        src_att = pickle.load(open(att_path, 'rb'))
        self.seen_wordvec = torch.from_numpy(normalize(src_att[0]))
        self.unseen_wordvec = torch.from_numpy(normalize(src_att[1][self.idx_top_unseen, :]))
        self.all_word_vec = torch.cat([self.seen_wordvec, self.unseen_wordvec], 0)
        self.class_name = labelset["all_label"]
        self.seen_class_name = labelset["seen_label_set"]
        self.unseen_class_name = labelset["unseen_label_set"]
        self.templates = [
            "a photo of the {}."
        ]

    def zeroshot_classifier(self, classnames, templates):
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            # texts = clip.tokenize(texts).cuda()  # tokenize
            # class_embeddings = self.model.clip_model.encode_text(texts)  # embed with text encoder
            texts = self.model.tokenizer(texts, padding=True, return_tensors="pt")  # tokenize
            texts = tuple([texts['input_ids'].cuda(), texts['attention_mask'].cuda()])
            class_embeddings = self.model.text_encoder(*texts).pooler_output  # embed with text encoder
            # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def _evaluate(self, mode='test_zsl'):
        self.model.eval()
        labs = torch.tensor([], dtype=torch.float).cuda()
        logits = torch.tensor([], dtype=torch.float).cuda()
        tqdm_obj = tqdm(self.dataloder[mode], ncols=80)
        if mode == 'test_zsl':
            zeroshot_weights = self.zeroshot_classifier(self.unseen_class_name,
                                                        self.templates).cuda()  # self._getprompt(self.nus_class_name_81, self.templates)
            vecs = self.unseen_wordvec
        else:
            zeroshot_weights = self.zeroshot_classifier(self.class_name,
                                                        self.templates).cuda()  # self._getprompt(self.nus_class_name_1006, self.templates)
            vecs = self.all_word_vec
        for step, batch in enumerate(tqdm_obj):
            img, lab = batch[0].cuda(), batch[1].cuda()
            with torch.no_grad():
                logit = self.model(img.float().cuda(), zeroshot_weights, vecs.cuda(), lab.cuda(), Train=False)
            labs = torch.cat([labs, lab])
            logits = torch.cat([logits, logit])
        # print(("completed calculating predictions over all {} images".format(c)))
        # logits = logits.clone()
        # sig_logits = torch.sigmoid(logits)
        map = util.compute_AP(logits.cuda(), labs.cuda())
        F1_3, P_3, R_3 = util.compute_F1(logits.cuda(), labs.cuda(), 'overall', k_val=10)
        F1_5, P_5, R_5 = util.compute_F1(logits.cuda(), labs.cuda(), 'overall', k_val=20)
        if mode == 'test_zsl':
            print('ZSL AP_OP', torch.mean(map).item())
            print('k=3_OP', torch.mean(F1_3).item(), torch.mean(P_3).item(), torch.mean(R_3).item())
            print('k=5_OP', torch.mean(F1_5).item(), torch.mean(P_5).item(), torch.mean(R_5).item())
            wandb.log({
                "AP_ZSL_OP": torch.mean(map).item(),
                "F1_3_ZSL_OP": torch.mean(F1_3).item(),
                "P_3_ZSL_OP": torch.mean(P_3).item(),
                "R_3_ZSL_OP": torch.mean(R_3).item(),
                "F1_5_ZSL_OP": torch.mean(F1_5).item(),
                "P_5_ZSL_OP": torch.mean(P_5).item(),
                "R_5_ZSL_OP": torch.mean(R_5).item(),
            })
        else:
            print('GZSL AP_OP', torch.mean(map).item())
            print('k=3_OP', torch.mean(F1_3).item(), torch.mean(P_3).item(), torch.mean(R_3).item())
            print('k=5_OP', torch.mean(F1_5).item(), torch.mean(P_5).item(), torch.mean(R_5).item())
            wandb.log({
                "AP_GZSL_OP": torch.mean(map).item(),
                "F1_3_GZSL_OP": torch.mean(F1_3).item(),
                "P_3_GZSL_OP": torch.mean(P_3).item(),
                "R_3_GZSL_OP": torch.mean(R_3).item(),
                "F1_5_GZSL_OP": torch.mean(F1_5).item(),
                "P_5_GZSL_OP": torch.mean(P_5).item(),
                "R_5_GZSL_OP": torch.mean(R_5).item(),
            })
        return torch.mean(map).item()

    def epoch_evaluate(self):
        print(">>>>>>>>>>>>>Testing")
        ap_zsl = self._evaluate(mode='test_zsl')
        ap_gzsl = self._evaluate(mode='test_gzsl')
        return ap_zsl


def main_fun():
    """Main method for training.
    Args:
        distributed: if distributed train.
    """
    # 0. Load config and mkdir
    wandb.config.update(opt)
    seed = opt.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    # wandb.config.update(opt)
    # 1. Load data
    print(">>>>>>>>>>>>>Load Data")
    path = 'datasets/OpenImages/2018_04/'
    seen_labelmap_path = path + '/classes-trainable.txt'
    dict_path = path + '/class-descriptions.csv'
    unseen_labelmap_path = path + '/unseen_labels.pkl'
    seen_label_set, unseen_label_set_all, label_dict = dataset.LoadLabelMap(seen_labelmap_path, unseen_labelmap_path,
                                                                            dict_path)
    path_top_unseen = os.path.join('datasets/OpenImages/2018_04/', 'top_400_unseen.csv')
    df_top_unseen = pd.read_csv(path_top_unseen, header=None)
    idx_top_unseen = df_top_unseen.values[:, 0]
    unseen_label_set = []
    for i in range(len(idx_top_unseen)):
        unseen_label_set.append(unseen_label_set_all[idx_top_unseen[i]])
    all_label_set = seen_label_set + unseen_label_set
    # test_images = pd.read_csv(path + 'test/images.csv')['ImageID']
    # test_annot = pd.read_csv(path + 'test/test-annotations-human.csv')
    train_annot = pd.read_csv(path + 'my_train/train-annotations-human.csv')
    val_annot = pd.read_csv(path + 'my_val/val-annotations-human.csv')
    labelset = {"seen_label_set": seen_label_set, "unseen_label_set": unseen_label_set, "all_label": all_label_set}
    # labelset = {"seen_label_set": seen_label_set, "unseen_label_set": unseen_label_set, "all_label": all_label_set}
    test_zsl_dataset = dataset.OpenimagesDataset('/home/ttl/dataset/openimages/images',
                                                 val_annot,
                                                 transform=dataset.transform_test, label_set=unseen_label_set,
                                                 mode='val')
    print("len(test_zsl_dataset_zsl)", len(test_zsl_dataset))
    test_zsl_loader = DataLoader(dataset=test_zsl_dataset, batch_size=512, shuffle=False,
                                 num_workers=opt.workers, drop_last=True, pin_memory=False)
    test_gzsl_dataset = dataset.OpenimagesDataset('/home/ttl/dataset/openimages/images',
                                                  val_annot,
                                                  transform=dataset.transform_test, label_set=all_label_set, mode='val')
    print("len(test_gzsl_dataset)", len(test_gzsl_dataset))
    test_gzsl_loader = DataLoader(dataset=test_gzsl_dataset, batch_size=8, shuffle=False,
                                  num_workers=opt.workers, drop_last=True, pin_memory=False)
    dataloader = {
        'test_zsl': test_zsl_loader,
        'test_gzsl': test_gzsl_loader,
    }

    # 2. Build model
    print(">>>>>>>>>>>>>Build model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mymodel.Mymodel(model_name=opt.model_name, opt=opt)  # clip.load("RN50", device=device)  # ViT-B/32
    state_dict = torch.load('models/model_openimages_epoch0.bin')
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    # 3. Train and Save model
    # print(">>>>>>>>>>>>>Training")
    testt = Test(model=model, dataloader=dataloader,device=device, config=opt, labelset=labelset)
    best_dev_map = testt.epoch_evaluate()
    print(best_dev_map)



if __name__ == '__main__':
    # torch.cuda.set_device(1)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', default=0, help='used for distributed parallel')
    # parser.add_argument("--distributed", action="store_true", help="if distributed train.")
    # parser.add_argument("--block_num", type=int, default=0, help="block num")
    # parser.add_argument("--block_size", type=int, default=0, help="block size")
    # parser.add_argument("--temp", type=float, default=0.3, help="block size")
    # args = parser.parse_args()
    main_fun()
