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

import dataset
from torch.utils.data import DataLoader
# from warmup_scheduler import GradualWarmupScheduler
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup)
from torch.cuda.amp import autocast, GradScaler
import wandb
import time
from sklearn.preprocessing import normalize
import torch.nn.functional as F
# from ranger21 import Ranger21
wandb.init(project='mlc_zsl', entity='shilida')
from transformers import CLIPTokenizer
import clip


class Trainer:
    def __init__(self,
                 model, dataloader, device, config, distributed):
        self.model = model
        self.device = device
        self.config = config
        self.dataloder = dataloader
        self.num_training_steps = config.nepoch * len(dataloader['train'])
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.distributed = distributed
        self.file_tag1k = 'datasets/NUS-WIDE/NUS_WID_Tags/TagList1k.txt'
        self.file_tag81 = 'datasets/NUS-WIDE/ConceptsList/Concepts81.txt'
        att_path = 'datasets/NUS-WIDE/wiki_contexts/NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300'
        self.seen_cls_idx, _ = self.get_seen_unseen_classes()
        src_att = pickle.load(open(att_path, 'rb'))
        self.vecs_925 = torch.from_numpy(normalize(src_att[0][self.seen_cls_idx])).cuda()  # (925,300)
        self.vecs_81 = torch.from_numpy(normalize(src_att[1])).cuda()  # (81,300)
        self.vecs_1006 = torch.cat([self.vecs_925, self.vecs_81], 0).cuda()
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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
        self.nus_class_name_1006, self.nus_class_name_81, self.nus_class_name_925 = self._get_classes()

    def zeroshot_classifier(self, classnames, templates):
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            # texts = clip.tokenize(texts).cuda()  # tokenize
            # class_embeddings = self.model.clip_model.encode_text(texts)  # embed with text encoder
            texts = self.model.tokenizer(texts, padding=True, return_tensors="pt")  # tokenize
            texts = tuple([texts['input_ids'].cuda(), texts['attention_mask'].cuda()])    # tuple：元组
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

    # def zeroshot_classifier_single(self, classnames, templates, labels):
    #     zeroshot_weights = []
    #     bs = labels.shape[0]
    #     for j in range(bs):
    #         for i in range(len(classnames)):
    #             if labels[j][i] != 1:
    #                 continue
    #             else:
    #                 texts = [template.format(classnames[i]) for template in templates]  # format with class
    #                 texts = clip.tokenize(texts).cuda()  # tokenize
    #                 class_embeddings = self.model.clip_model.encode_text(texts)  # embed with text encoder
    #                 class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #                 class_embedding = class_embeddings.mean(dim=0)
    #                 class_embedding /= class_embedding.norm()
    #                 zeroshot_weights.append(class_embedding)
    #     zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    #     return zeroshot_weights

    def get_seen_unseen_classes(self):
        with open(self.file_tag1k, "r") as file:
            tag1k = np.array(file.read().splitlines())
        with open(self.file_tag81, "r") as file:
            tag81 = np.array(file.read().splitlines())
        seen_cls_idx = np.array(
            [i for i in range(len(tag1k)) if tag1k[i] not in tag81])
        unseen_cls_idx = np.array(
            [i for i in range(len(tag1k)) if tag1k[i] in tag81])
        return seen_cls_idx, unseen_cls_idx  # seen_cls_idx 925   unseen_cls_idx75

    def _get_classes(self):
        with open(self.file_tag1k, "r") as file:
            tag1k = np.array(file.read().splitlines())
        with open(self.file_tag81, "r") as file:
            tag81 = np.array(file.read().splitlines())
        tag1k = list(tag1k)
        tag81 = list(tag81)
        tag925 = list(set(tag1k).difference(set(tag81)))     #list差集。tag1k中有而tag81没有
        tag1k.extend(tag81)
        tag1006 = set(tag1k)
        return tag1006, tag81, tag925

    # def get_text(self, classnames, templates):
    #     all_text = []
    #     for classname in classnames:
    #         texts = [template.format(classname) for template in templates]
    #         print("texts",texts)
    #         all_text.extend(texts)
    #     print("all_text",all_text)
    #     all_text = clip.tokenize(all_text).cuda()
    #     print('all_text.shape',all_text.shape)
    #     return all_text
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
        t = torch.gt(label,0).cpu()
        label_classname =[]
        for i in range(label.shape[0]):
            label_classname.append(nus_class_name_925[t[i]])
        label_prompts = []
        for classnames in label_classname:
            label_prompts.append([self.templates[0].format(classname) for classname in classnames])#template.format(classname)self.templates[0]#[template.format(classname) for template in self.templates]
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
            num_warmup_steps=self.config.num_warmup_steps * self.num_training_steps, #num_warmup_steps:初始预热步数
            num_training_steps=self.num_training_steps) # num_training_steps：整个训练过程的总步数
        # scheduler = get_constant_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=self.config.num_warmup_steps * self.num_training_steps)
        return scheduler
    def _getprompt(self,classnames, templates):
        texts = []
        for classname in classnames:
            text = templates[0].format(classname) # format with class
            texts.append(text)
        return texts
    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)

    def train(self):
        scaler = GradScaler()   #在训练开始之前创建依次Scaler
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[0, 1], find_unused_parameters=True)
        trange_obj = trange(self.config.nepoch, desc='Epoch', ncols=120)
        best_map = 0
        best_model_state_dict, best_zsl_map, best_gzsl_map = None, 0, 0
        zeroshot_weights = self.zeroshot_classifier(self.nus_class_name_925, self.templates).cuda()
        for epoch, _ in enumerate(trange_obj):
            self.model.train()
            tqdm_obj = tqdm(self.dataloder['train'], ncols=80)
            count = 0
            loss_avg = 0
            # zeroshot_weights = self._getprompt(self.nus_class_name_925, self.templates)
            for step, batch in enumerate(tqdm_obj):#现在在这里呢，所以要想跳出来可以直接点下一行。  看好，要进入循环了
                self.optimizer.zero_grad()#点一下这一行，其实是上面这个命令下的随便一行就可以
                _, img, labels = batch[0], batch[1].cuda(), batch[2].cuda()
                # labels = torch.clamp(labels, 0, 1)
                # loss = self.model(img, zeroshot_weights, self.vecs_925, labels, Train=True)
                with autocast():    #使用自动转换功能前向计算，会自适应将FP32转FP16
                    labels = torch.clamp(labels, 0, 1)
                    # temp_seen_labels = labels.sum(1)
                    # labels = labels[temp_seen_labels > 0]
                    # img = img[temp_seen_labels > 0]
                    # label_prompts = self._trans_label_to_prompt(labels)
                    # text_featrue = self. _label_prompt_encoder(label_prompts).cuda()
                    # print('text_featrue.shape',text_featrue.shape)
                    # inputs = self.model.processor(text=zeroshot_weights, images=img.cpu(), return_tensors="pt",padding=True)
                    # loss = self.model(inputs,labels)
                    loss = self.model(img, zeroshot_weights, self.vecs_925, labels, Train=True)
                    # logits_925, zsl_word_vec = self.model(img, text, self.vecs_925)
                    # loss_mse = self.mse(zsl_word_vec, zeroshot_weights.transpose(0, 1))
                    # loss_ce = self.criterion(logits_925, labels.float().cuda())
                    # loss = 0.9 * loss_mse + loss_ce
                scaler.scale(loss).backward()  #首先进行损失缩放. 调用backward()在缩放后的损失上创建一个缩放的梯度
                self.scheduler.step()
                scaler.step(self.optimizer)   #还原先前的缩放
                scaler.update()       # 为下一次迭代更新缩放器
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
        return best_model_state_dict, best_map

    def _epoch_evaluate(self):
        print(">>>>>>>>>>>>>Testing")
        self.model.eval()
        lab_1006_all = torch.tensor([], dtype=torch.float).cuda()
        lab_81_all = torch.tensor([], dtype=torch.float).cuda()
        logits_81_all = torch.tensor([], dtype=torch.float).cuda()
        logits_1006_all = torch.tensor([], dtype=torch.float).cuda()
        tqdm_obj = tqdm(self.dataloder['test'], ncols=80)
        zeroshot_weights_81 = self.zeroshot_classifier(self.nus_class_name_81, self.templates).cuda()#self._getprompt(self.nus_class_name_81, self.templates)
        zeroshot_weights_1006 = self.zeroshot_classifier(self.nus_class_name_1006, self.templates).cuda()#self._getprompt(self.nus_class_name_1006, self.templates)
        for step, batch in enumerate(tqdm_obj):
            _, img, lab_1006, lab_81 = batch[0], batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            with torch.no_grad():
                logits_81 = self.model(img.float(), zeroshot_weights_81, self.vecs_81.float(), lab_81,
                                       Train=False)
                logits_1006 = self.model(img.float(), zeroshot_weights_1006, self.vecs_1006.float(), lab_1006,
                                         Train=False)
            lab_1006_all = torch.cat([lab_1006_all, lab_1006])
            lab_81_all = torch.cat([lab_81_all, lab_81])
            logits_81_all = torch.cat([logits_81_all, logits_81])
            logits_1006_all = torch.cat([logits_1006_all, logits_1006])

        # print(("completed calculating predictions over all {} images".format(c)))
        logits_81_5 = logits_81_all.clone()
        ap_81 = util.compute_AP(logits_81_all.cuda(), lab_81_all.cuda())
        F1_3_81, P_3_81, R_3_81 = util.compute_F1(logits_81_all.cuda(), lab_81_all.cuda(), 'overall', k_val=3)
        F1_5_81, P_5_81, R_5_81 = util.compute_F1(logits_81_5.cuda(), lab_81_all.cuda(), 'overall', k_val=5)
        print('ZSL AP', torch.mean(ap_81).item())
        print('k=3', torch.mean(F1_3_81).item(), torch.mean(P_3_81).item(), torch.mean(R_3_81).item())
        print('k=5', torch.mean(F1_5_81).item(), torch.mean(P_5_81).item(), torch.mean(R_5_81).item())
        logits_1006_5 = logits_1006_all.clone()
        ap_1006 = util.compute_AP(logits_1006_all.cuda(), lab_1006_all.cuda())
        F1_3_1006, P_3_1006, R_3_1006 = util.compute_F1(logits_1006_all.cuda(), lab_1006_all.cuda(), 'overall', k_val=3)
        F1_5_1006, P_5_1006, R_5_1006 = util.compute_F1(logits_1006_5.cuda(), lab_1006_all.cuda(), 'overall', k_val=5)

        print('GZSL AP', torch.mean(ap_1006).item())
        print('g_k=3', torch.mean(F1_3_1006).item(), torch.mean(P_3_1006).item(), torch.mean(R_3_1006).item())
        print('g_k=5', torch.mean(F1_5_1006).item(), torch.mean(P_5_1006).item(), torch.mean(R_5_1006).item())
        wandb.log({
            "AP_ZSL": torch.mean(ap_81).item(),
            "F1_3_81": torch.mean(F1_3_81).item(),
            "P_3_81": torch.mean(P_3_81).item(),
            "R_3_81": torch.mean(R_3_81).item(),
            "F1_5_81": torch.mean(F1_5_81).item(),
            "P_5_81": torch.mean(P_5_81).item(),
            "R_5_81": torch.mean(R_5_81).item(),
            "AP_GZSL": torch.mean(ap_1006).item(),
            "F1_3_1006": torch.mean(F1_3_1006).item(),
            "P_3_1006": torch.mean(P_3_1006).item(),
            "R_3_1006": torch.mean(R_3_1006).item(),
            "F1_5_1006": torch.mean(F1_5_1006).item(),
            "P_5_1006": torch.mean(P_5_1006).item(),
            "R_5_1006": +torch.mean(R_5_1006).item()
        })
        return torch.mean(ap_81).item()


def main_fun():
    """Main method for training.
    Args:
        distributed: if distributed train.
    """
    # 0. Load config and mkdir
    wandb.config.update(opt)  #wandb可视化平台
    seed = opt.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    # wandb.config.update(config)
    # 1. Load data
    print(">>>>>>>>>>>>>Load Data")
    src = 'datasets/NUS-WIDE/features/'
    train_dataset = dataset.get_extract_data(
        dir_=os.path.join('/home/ttl/dataset/NUS-WIDE/nuswide/Flickr/'),
        # /root/mlc-zsl/datasets/NUS-WIDE/nuswide/Flickr/
        json_file=os.path.join(src, 'nus_wide_train.json'), type='Train')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.workers, drop_last=False, pin_memory=False)   #用于设置有多少个子进程负责数据加载
    test_dataset = dataset.get_extract_data(
        dir_=os.path.join('/home/ttl/dataset/NUS-WIDE/nuswide/Flickr/'),
        json_file=os.path.join(src, 'nus_wide_test.json'), type='Test')
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=opt.val_batch_size, shuffle=False,
                                  num_workers=opt.workers, drop_last=False, pin_memory=False)
    dataloader = {
        'train': train_data_loader,
        'test': test_data_loader}
    # 2. Build model
    print(">>>>>>>>>>>>>Build model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mymodel.Mymodel(model_name=opt.model_name,opt=opt)  # clip.load("RN50", device=device)  # ViT-B/32
    model.to(device)
    # 3. Train and Save model
    print(">>>>>>>>>>>>>Training")
    trainer = Trainer(model=model, dataloader=dataloader,
                      device=device, config=opt, distributed=False)
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
