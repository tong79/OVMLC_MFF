import os
from myconfig import opt

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(opt.gpu_id)
import torch
import mymodel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import model as model
import util_nus as util

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

class Trainer:
    def __init__(self,
                 model, dataloader, device, config, distributed, labelset):
        self.model = model
        self.device = device
        self.config = config
        self.dataloder = dataloader
        self.num_training_steps = config.nepoch * len(dataloader['train'])
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.distributed = distributed
        # with open('datasets/COCO/wordvec_array.pickle', 'rb') as f:
        #     glove = pickle.load(f)
        # with open('datasets/COCO/cls_ids.pickle', 'rb') as f:
        #     cls_ids = pickle.load(f)
        # seen_ids = cls_ids['train']
        # unseen_ids = cls_ids['test']
        # wordvec_array = glove['wordvec_array']
        # seen_wordvec = deepcopy(wordvec_array)
        # self.seen_wordvec = torch.tensor(seen_wordvec[:, :, list(seen_ids)]).squeeze()
        # unseen_wordvec = deepcopy(wordvec_array)
        # self.unseen_wordvec = torch.tensor(unseen_wordvec[:, :, list(unseen_ids)]).squeeze()
        # self.all_wordvec = torch.tensor(wordvec_array[:, :, list(seen_ids | unseen_ids)]).squeeze()
        # path = os.path.join('datasets/OpenImages', '2018_04')
        att_path = os.path.join('datasets/OpenImages/wiki_contexts/', 'OpenImage_w2v_context_window_10_glove-wiki-gigaword-300.pkl')
        path_top_unseen = os.path.join('datasets/OpenImages/2018_04/', 'top_400_unseen.csv')
        df_top_unseen = pd.read_csv(path_top_unseen, header=None)
        self.idx_top_unseen = df_top_unseen.values[:, 0]
        assert len(self.idx_top_unseen) == 400
        src_att = pickle.load(open(att_path, 'rb'))
        self.seen_wordvec = torch.from_numpy(normalize(src_att[0]))
        self.unseen_wordvec = torch.from_numpy(normalize(src_att[1][self.idx_top_unseen, :]))
        self.all_word_vec = torch.cat([self.seen_wordvec, self.unseen_wordvec], 0)
        # num_seen = 65
        # all_word_vec = np.loadtxt('datasets/COCO/word_glo.txt', dtype='float32', delimiter=',')
        # seen_wordvec = all_word_vec[:, :num_seen]
        # unseen_wordvec = all_word_vec[:, num_seen:]
        # self.seen_wordvec = torch.tensor(seen_wordvec).transpose(0, 1)
        # self.unseen_wordvec = torch.tensor(unseen_wordvec).transpose(0, 1)
        # self.all_word_vec = torch.tensor(all_word_vec).transpose(0, 1)
        self.class_name = labelset["all_label"]
        self.seen_class_name = labelset["seen_label_set"]
        self.unseen_class_name = labelset["unseen_label_set"]
        self.templates = [
            "a photo of the {}.",
            # "a photo contains the {}.",
            # "a good photo contains the big {}.",
            # "a good photo contains the small {}.",
            # "a good photo contraining the {}.",
            # "itap of a {}.",
            # "a bad photo of the {}.",
            # "a origami {}.",
            # "a photo of the large {}.",
            # "a {} in a video game.",
            # "art of the {}.",
            # "a photo of the small {}.",
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

    def _label_prompt_encoder(self, label_prompt):
        text_embedding = []
        for text in label_prompt:
            texts = self.model.tokenizer(text, padding=True, return_tensors="pt")  # tokenize
            texts = tuple([texts['input_ids'].cuda(), texts['attention_mask'].cuda()])
            class_embeddings = self.model.text_encoder(*texts).pooler_output
            text_embedding.append(class_embeddings)
            # for classname in classnames:
            #     texts = [template.format(classname) for template in templates]  # format with class
            #     # texts = clip.tokenize(texts).cuda()  # tokenize
            #     # class_embeddings = self.model.clip_model.encode_text(texts)  # embed with text encoder
            #     texts = self.model.tokenizer(texts, padding=True, return_tensors="pt")  # tokenize
            #     texts = tuple([texts['input_ids'].cuda(), texts['attention_mask'].cuda()])
            #     class_embeddings = self.model.text_encoder(*texts).pooler_output  # embed with text encoder
            #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            #     class_embedding = class_embeddings.mean(dim=0)
            #     class_embedding /= class_embedding.norm()
            #     text_embedding.append(class_embedding)
        text_embedding = torch.nn.utils.rnn.pad_sequence(text_embedding)
        # text_embedding = torch.stack(text_embedding, dim=1).cuda()
        return text_embedding

    def _get_optimizer(self):
        """Get optimizer for different models.
        Returns:
            optimizer
        """
        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters()
        #                 if not any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in self.model.named_parameters()
        #                 if any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.0},
        # ]
        optimizer_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = Ranger21(
        #     optimizer_parameters, lr=self.config.lr, num_epochs=self.config.nepoch, num_batches_per_epoch=self.config.batch_size)
        # optimizer_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(
            optimizer_parameters,
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-8,
            correct_bias=False)
        return optimizer

    def _trans_label_to_prompt(self, label):
        nus_class_name_925 = np.array(self.nus_class_name_925)
        t = torch.gt(label, 0).cpu()
        label_classname = []
        for i in range(label.shape[0]):
            label_classname.append(nus_class_name_925[t[i]])
        label_prompts = []
        for classnames in label_classname:
            label_prompts.append([self.templates[0].format(classname) for classname in
                                  classnames])  # template.format(classname)self.templates[0]#[template.format(classname) for template in self.templates]
        # print(label_prompts)
        return label_prompts

    def _get_scheduler(self):
        """Get scheduler for different models.
        Returns:
            scheduler
        """
        # get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps * self.num_training_steps,
            num_training_steps=self.num_training_steps)
        # scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.lr, steps_per_epoch=len(self.dataloder['train']), epochs=self.config.nepoch,
        #                                 pct_start=0.2)
        # scheduler = get_constant_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=self.config.num_warmup_steps * self.num_training_steps)
        return scheduler

    def _getprompt(self, classnames, templates):
        texts = []
        for classname in classnames:
            text = templates[0].format(classname)  # format with class
            texts.append(text)
        return texts

    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)

    def train(self):
        scaler = GradScaler()
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[0, 1], find_unused_parameters=True)
        trange_obj = trange(self.config.nepoch, desc='Epoch', ncols=120)
        best_map = 0
        best_model_state_dict, best_zsl_map, best_gzsl_map = None, 0, 0
        zeroshot_weights = self.zeroshot_classifier(self.seen_class_name, self.templates).cuda()
        for epoch, _ in enumerate(trange_obj):
            self.model.train()
            tqdm_obj = tqdm(self.dataloder['train'], ncols=80)
            count = 0
            loss_avg = 0
            # zeroshot_weights = self._getprompt(self.nus_class_name_925, self.templates)
            for step, batch in enumerate(tqdm_obj):
                self.optimizer.zero_grad()
                img, labels = batch[0], batch[1].cuda()
                with autocast():
                    labels = torch.clamp(labels, 0, 1)
                    # temp_label = torch.sum(labels > 0,1) > 0  # remove those images that don not have even a single 1 (positive label).
                    # train_labels = labels[temp_label]
                    # train_inputs = train_inputs[temp_label]
                    # _train_labels = train_labels[torch.clamp(train_labels, 0, 1).sum(1) <= 40]
                    # train_inputs = train_inputs[torch.clamp(train_labels, 0, 1).sum(1) <= 40]
                    # train_inputs = train_inputs.cuda()
                    # _train_labels = _train_labels.cuda()
                    loss = self.model(img.cuda(), zeroshot_weights, self.seen_wordvec.cuda(), labels, Train=True)
                    # print(loss)
                    # logits_925, zsl_word_vec = self.model(img, text, self.vecs_925)
                    # loss_mse = self.mse(zsl_word_vec, zeroshot_weights.transpose(0, 1))
                    # loss_ce = self.criterion(logits_925, labels.float().cuda())
                    # loss = 0.9 * loss_mse + loss_ce
                scaler.scale(loss).backward()
                self.scheduler.step()
                scaler.step(self.optimizer)
                scaler.update()
                # 如果不使用双精度
                # loss.backward()
                # self.scheduler.step()
                # self.optimizer.step()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                count = count + 1
                loss_avg = loss_avg + loss
                if count % 10 == 0:
                    loss_avg = loss_avg / 10
                    wandb.log({
                        "loss": loss_avg,
                        "lr": lr
                    })
                    loss_avg = 0
                    count = 0
            results = self._epoch_evaluate()
            if results > best_map:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_map = results
            torch.save(self.model.state_dict(),
                       os.path.join(opt.save_path, 'model_openimages_{}.bin'.format(results)))
        return best_model_state_dict, best_map

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

    def _epoch_evaluate(self):
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
    # all_label_set = []
    # seen = 65
    # wordname_all = open('datasets/COCO/cls_names_test_coco.csv').read().split("\n")
    # for idx in range(int(len(wordname_all)) - 1):
    #     all_label_set.append(wordname_all[idx].split(',')[0])
    # seen_label_set = all_label_set[:seen]
    # unseen_label_set = all_label_set[seen:]
    # with open(r"datasets/COCO/seen_classes.txt", "r", encoding="utf-8") as f:
    #     text = f.readlines()
    # seen_label_set = [line.strip("\n") for line in text]
    # with open(r"datasets/COCO/unseen_classes.txt", "r", encoding="utf-8") as f:
    #     text = f.readlines()
    # unseen_label_set = [line.strip("\n") for line in text]
    # all_label = seen_label_set + unseen_label_set
    path = 'datasets/OpenImages/2018_04/'
    seen_labelmap_path = path + '/classes-trainable.txt'
    dict_path = path + '/class-descriptions.csv'
    unseen_labelmap_path = path + '/unseen_labels.pkl'
    seen_label_set, unseen_label_set_all, label_dict = dataset.LoadLabelMap(seen_labelmap_path, unseen_labelmap_path, dict_path)
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
    train_dataset = dataset.OpenimagesDataset('/home/ttl/dataset/openimages/images',
                                        train_annot,
                                        transform=dataset.transform, label_set=seen_label_set,mode='train')
    print("len(train_dataset)", len(train_dataset))
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.workers, drop_last=True, pin_memory=False)
    test_zsl_dataset = dataset.OpenimagesDataset('/home/ttl/dataset/openimages/images',
                                           val_annot,
                                           transform=dataset.transform_test, label_set=unseen_label_set,mode='val')
    print("len(test_zsl_dataset_zsl)", len(test_zsl_dataset))
    test_zsl_loader = DataLoader(dataset=test_zsl_dataset, batch_size=opt.batch_size, shuffle=False,
                                 num_workers=opt.workers, drop_last=True, pin_memory=False)
    test_gzsl_dataset = dataset.OpenimagesDataset('/home/ttl/dataset/openimages/images',
                                           val_annot,
                                            transform=dataset.transform_test, label_set=all_label_set,mode='val')
    print("len(test_gzsl_dataset)", len(test_gzsl_dataset))
    test_gzsl_loader = DataLoader(dataset=test_gzsl_dataset, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.workers, drop_last=True, pin_memory=False)
    dataloader = {
        'train': train_data_loader,
        'test_zsl': test_zsl_loader,
        'test_gzsl': test_gzsl_loader,
    }

    # 2. Build model
    print(">>>>>>>>>>>>>Build model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mymodel.Mymodel(model_name=opt.model_name, opt=opt)  # clip.load("RN50", device=device)  # ViT-B/32
    model.to(device)
    # 3. Train and Save model
    print(">>>>>>>>>>>>>Training")
    trainer = Trainer(model=model, dataloader=dataloader,
                      device=device, config=opt, distributed=False, labelset=labelset)
    best_model_dev_dict, best_dev_map = trainer.train()
    # 4. Save model
    torch.save(best_model_dev_dict,
               os.path.join(opt.save_path, 'model_{}.bin'.format(best_dev_map)))


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
