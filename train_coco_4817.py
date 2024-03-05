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
wandb.init(project='mlc_zsl', entity='shilida')
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
        with open('datasets/COCO/wordvec_array.pickle', 'rb') as f:
            glove = pickle.load(f)
        with open('datasets/COCO/cls_ids.pickle', 'rb') as f:
            cls_ids = pickle.load(f)
        seen_ids = cls_ids['train']
        unseen_ids = cls_ids['test']
        wordvec_array = glove['wordvec_array']
        seen_wordvec = deepcopy(wordvec_array)
        self.seen_wordvec = torch.tensor(seen_wordvec[:, :, list(seen_ids)]).squeeze().transpose(0,1)
        unseen_wordvec = deepcopy(wordvec_array)
        self.unseen_wordvec = torch.tensor(unseen_wordvec[:, :, list(unseen_ids)]).squeeze().transpose(0,1)
        self.all_word_vec = torch.tensor(wordvec_array[:, :, list(seen_ids | unseen_ids)]).squeeze().transpose(0,1)
        self.coco_class_name = labelset["all_label"]
        self.coco_seen_class_name = labelset["seen_label_set"]
        self.coco_unseen_class_name = labelset["unseen_label_set"]
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
        # scheduler = get_constant_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=self.config.num_warmup_steps * self.num_training_steps)
        return scheduler

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
        zeroshot_weights = self.zeroshot_classifier(self.coco_seen_class_name, self.templates).cuda()
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
        return best_model_state_dict, best_map

    def _evaluate(self, mode='test_zsl'):
        self.model.eval()
        labs = torch.tensor([], dtype=torch.float).cuda()
        logits = torch.tensor([], dtype=torch.float).cuda()
        tqdm_obj = tqdm(self.dataloder[mode], ncols=80)
        if mode == 'test_zsl':
            zeroshot_weights = self.zeroshot_classifier(self.coco_unseen_class_name,
                                                        self.templates).cuda()  # self._getprompt(self.nus_class_name_81, self.templates)
            vecs = self.unseen_wordvec
        else:
            zeroshot_weights = self.zeroshot_classifier(self.coco_class_name,
                                                        self.templates).cuda()  # self._getprompt(self.nus_class_name_1006, self.templates)
            vecs = self.all_word_vec
        for step, batch in enumerate(tqdm_obj):
            img, lab = batch[0].cuda(), batch[1].cuda()
            with torch.no_grad():
                logit = self.model(img.float().cuda(), zeroshot_weights, vecs.cuda(), lab.cuda(), Train=False)
            labs = torch.cat([labs, lab])
            # print(logit.shape)
            logits = torch.cat([logits, logit])
        # print(("completed calculating predictions over all {} images".format(c)))
        logits = torch.sigmoid(logits)
        logits = logits.clone()
        # sig_logits = torch.sigmoid(logits)
        map = util.compute_AP(logits.cuda(), labs.cuda())
        F1_3, P_3, R_3 = util.compute_F1(logits.cuda(), labs.cuda(), 'overall', k_val=3)
        F1_5, P_5, R_5 = util.compute_F1(logits.cuda(), labs.cuda(), 'overall', k_val=5)
        if mode == 'test_zsl':
            print('ZSL AP_COCO', torch.mean(map).item())
            print('k=3_COCO', torch.mean(F1_3).item(), torch.mean(P_3).item(), torch.mean(R_3).item())
            print('k=5_COCO', torch.mean(F1_5).item(), torch.mean(P_5).item(), torch.mean(R_5).item())
            wandb.log({
                "AP_ZSL_COCO": torch.mean(map).item(),
                "F1_3_ZSL_COCO": torch.mean(F1_3).item(),
                "P_3_ZSL_COCO": torch.mean(P_3).item(),
                "R_3_ZSL_COCO": torch.mean(R_3).item(),
                "F1_5_ZSL_COCO": torch.mean(F1_5).item(),
                "P_5_ZSL_COCO": torch.mean(P_5).item(),
                "R_5_ZSL_COCO": torch.mean(R_5).item(),
            })
        else:
            print('GZSL AP_COCO', torch.mean(map).item())
            print('k=3_COCO', torch.mean(F1_3).item(), torch.mean(P_3).item(), torch.mean(R_3).item())
            print('k=5_COCO', torch.mean(F1_5).item(), torch.mean(P_5).item(), torch.mean(R_5).item())
            wandb.log({
                "AP_GZSL_COCO": torch.mean(map).item(),
                "F1_3_GZSL_COCO": torch.mean(F1_3).item(),
                "P_3_GZSL_COCO": torch.mean(P_3).item(),
                "R_3_GZSL_COCO": torch.mean(R_3).item(),
                "F1_5_GZSL_COCO": torch.mean(F1_5).item(),
                "P_5_GZSL_COCO": torch.mean(P_5).item(),
                "R_5_GZSL_COCO": torch.mean(R_5).item(),
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
    with open(r"datasets/COCO/48_seen_classes.txt", "r", encoding="utf-8") as f:
        text = f.readlines()
    seen_label_set = [line.strip("\n") for line in text]
    with open(r"datasets/COCO/17_unseen_classes.txt", "r", encoding="utf-8") as f:
        text = f.readlines()
    unseen_label_set = [line.strip("\n") for line in text]
    all_label_set = seen_label_set + unseen_label_set
    labelset = {"seen_label_set": seen_label_set, "unseen_label_set": unseen_label_set, "all_label": all_label_set}
    train_dataset = dataset.CocoDataset('/home/shilida/multilabel/COCO2014/train2014',
                                        '/home/shilida/multilabel/COCO2014/annotations/instances_train2014.json',mode="train",
                                        transform=dataset.transform, label_set=seen_label_set)
    print("len(train_dataset)",len(train_dataset))
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.workers, drop_last=False, pin_memory=False)
    test_zsl_dataset = dataset.CocoDataset('/home/shilida/multilabel/COCO2014/val2014',
                                           '/home/shilida/multilabel/COCO2014/annotations/instances_val2014.json',mode="val",
                                           transform=dataset.transform, label_set=unseen_label_set)
    print("len(test_zsl_dataset)", len(test_zsl_dataset))
    test_zsl_loader = DataLoader(dataset=test_zsl_dataset, batch_size=opt.batch_size, shuffle=False,
                                 num_workers=opt.workers, drop_last=False, pin_memory=False)
    test_gzsl_dataset = dataset.CocoDataset('/home/shilida/multilabel/COCO2014/val2014',
                                            '/home/shilida/multilabel/COCO2014/annotations/instances_val2014.json',mode="val",
                                            transform=dataset.transform, label_set=all_label_set)
    test_gzsl_loader = DataLoader(dataset=test_gzsl_dataset, batch_size=opt.batch_size, shuffle=False,
                                   num_workers=opt.workers, drop_last=False, pin_memory=False)
    print("len(test_gzsl_dataset)", len(test_gzsl_dataset))
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
