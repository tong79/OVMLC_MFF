import time

import torch
import torch.nn as nn
import json
from tqdm import tqdm
import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from tqdm import tqdm
import clip
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import CLIPProcessor, CLIPVisionModel
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class Clip_zsl_mlc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = torch.nn.Linear(300, 512)

    def forward(self, images, **kwargs):
        return self.model.visual(images, **kwargs)


class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        # self.l2 = nn.Sequential(
        #     nn.Linear(mid_dim, mid_dim),
        #     nn.BatchNorm1d(mid_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.l3 = nn.Sequential(
        #     nn.Linear(mid_dim, out_dim),
        #     # nn.BatchNorm1d(out_dim)
        # )

    def forward(self, x):
        x = self.l1(x.float())
        # x = self.l2(x)
        # x = self.l3(x)

        return x


class ProjectionMLP_label(nn.Module):
    def __init__(self, in_dim_text, in_dim_label, mid_dim, out_dim):
        super(ProjectionMLP_label, self).__init__()
        self.l1_1 = nn.Sequential(
            nn.Linear(in_dim_text, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l1_2 = nn.Sequential(
            nn.Linear(in_dim_label, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, text_x, label_x):
        text_x = self.l1_1(text_x)
        label_x = self.l1_2(label_x)
        x = self.l2(text_x + label_x)
        x = self.l3(x)
        return x


def Nceloss(img, word, label, lossfn):
    cosine = F.linear(F.normalize(img, p=2, dim=1), F.normalize(word, p=2, dim=1))
    logits = 10 * cosine
    # print("logits.shape",logits.shape)
    # print("label.shape", label.shape)
    return lossfn(logits, label.float())


from torch.nn.modules.transformer import _get_activation_fn


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, emb: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt_300 = tgt[0]  # 原来的300维度
        tgt_text = tgt[1]
        # print('tgt_300.shape',tgt_300.shape)
        # print('tgt_text.shape',tgt_text.shape)
        tgt_300 = tgt_300 + self.dropout1(tgt_300)
        tgt_300 = self.norm1(tgt_300)
        tgt_300 = self.multihead_attn_1(tgt_300, emb, emb)[0]
        tgt_300 = tgt_300 + self.dropout2(tgt_300)
        tgt_text = self.norm2(tgt_300)
        tgt_text = tgt_text + self.dropout1(tgt_text)
        tgt_text = self.norm1(tgt_text)
        tgt_text = self.multihead_attn_2(tgt_text, emb, emb)[0]
        tgt_text = tgt_text + self.dropout2(tgt_text)
        tgt_text = self.norm2(tgt_text)
        tgt = tgt_text + tgt_300
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt_3002 = self.linear2(self.dropout(self.activation(self.linear1(tgt_300))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class FineTuneCLIP(nn.Module):
    def __init__(self, model_name="ViT-B/16",
                 freeze=True):  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        super(FineTuneCLIP, self).__init__()
        self.clip_model, preprocess = clip.load(model_name)  # CLIPModel.from_pretrained(model_name)
        # print(self.clip_model)
        self.clip_model.visual.attnpool = torch.nn.Sequential()
        # self.clip_model.visual.layer4 = torch.nn.Sequential()
        if freeze:
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = False
        # newversion
        self.conv = nn.Conv2d(2048, 512, 1)
        # self.transformer = nn.Transformer(512, 8, 6, 6)
        # self.wordvec_proj = torch.nn.Linear(300, 2048)
        # self.row_embed = nn.Parameter(torch.rand(50, 256))
        # self.col_embed = nn.Parameter(torch.rand(50, 256))
        # self.fc = torch.nn.Linear(512, 1)
        # self.projection_label = ProjectionMLP_label(512, 300, 1024, 2048)
        decoder_layer = nn.TransformerDecoderLayer(d_model=2048,
                                                   nhead=8)  # TransformerDecoderLayerOptimal(d_model=2048,nhead=4)  # # nn.TransformerDecoderLayer(d_model=2048, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc = torch.nn.Linear(2048, 1)
        # self.multihead_attn_1 = nn.MultiheadAttention(1024, 4, dropout=0.1)
        self.wordvec_proj = torch.nn.Linear(300, 2048)
        # self.fc = torch.nn.Linear(2048, 1)
        # self.awl = AutomaticWeightedLoss(2).cuda()
        self.bce = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()
        # self.ce = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.text_proj = torch.nn.Linear(1024, 2048)

    def forward(self, image, text_features, word_vec, labels, Train=False):
        image_features = self.clip_model.encode_image(image)  # [bs,1024, 14,14]
        # print("image_features.shape",image_features.shape)
        bs = image_features.shape[0]
        if len(image_features.shape) == 4:  # [bs,2048, 7,7]
            image_features = image_features.flatten(2).transpose(1, 2)
        tgt = torch.nn.functional.relu(self.wordvec_proj(word_vec.cuda())).unsqueeze(1).expand(-1, bs,
                                                                                               -1)  # self.projection_label(text_features.transpose(0, 1), word_vec)
        # print("tgt.shape",tgt)
        # tgt = [label_features.unsqueeze(1).expand(-1, bs, -1), text_features]
        # emb = [image_features.transpose(0, 1).float(), text_features]
        h = self.decoder(tgt, image_features.transpose(0, 1).float())
        h = h.transpose(0, 1)
        logits = self.fc(h).squeeze()
        # h = self.conv(image_features.float())
        # H, W = h.shape[-2:]
        # pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        #                  self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        #                  ], dim=-1).flatten(0, 1).unsqueeze(1)
        # query = torch.nn.functional.relu(self.wordvec_proj(word_vec.cuda()))
        # query = query.unsqueeze(1).expand(-1, bs, -1)
        # trans_input = pos + h.flatten(2).permute(2, 0, 1)
        # h = self.transformer(trans_input, query)
        # h = h.transpose(0, 1)
        # logits = self.fc(h).squeeze()
        # print('h.shape', h.shape)
        # print('logits.shape',logits.shape)
        if Train:
            loss_bce = self.bce(logits, labels.float())
            # loss_contrastive = Nceloss(image_features,label_features,labels,self.bce)
            # loss = self.awl(loss_bce,loss_contrastive)
            return loss_bce
        else:
            return logits

        # if len(image_features.shape) == 4:  # [bs,2048, 7,7]
        #     image_features = image_features.flatten(2).transpose(1, 2)
        # bs = image_features.shape[0]
        # # print("text_features.shape", text_features.shape)
        # text_features = torch.nn.functional.relu(self.text_proj(text_features.transpose(0, 1).cuda()))
        # text_features = text_features.unsqueeze(1).expand(-1, bs, -1)
        # label_features = torch.nn.functional.relu(
        #     self.wordvec_proj(word_vec.cuda()))  # self.projection_label(text_features.transpose(0, 1), word_vec)
        # tgt = [label_features.unsqueeze(1).expand(-1, bs, -1), text_features]
        # # emb = [image_features.transpose(0, 1).float(), text_features]
        # h = self.decoder(tgt, image_features.transpose(0, 1).float())
        # h = h.transpose(0, 1)
        # logits = self.fc(h).squeeze()
        # if Train:
        #     loss_bce = self.bce(logits, labels.float())
        #     # loss_contrastive = Nceloss(image_features,label_features,labels,self.bce)
        #     # loss = self.awl(loss_bce,loss_contrastive)
        #     return loss_bce
        # else:
        #     return logits
        # return image_features, label_features


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
from vit_model import vit_base_patch32_224

class Mymodel(nn.Module):
    def __init__(self, model_name, opt,
                 freeze=True):  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        super(Mymodel, self).__init__()
        # self.CLIP = CLIPModel.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.img_encoder = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if freeze:
            for parameter in self.img_encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
        # self.projection_image = ProjectionMLP(512, 2048, 2048)
        # self.projection_label = ProjectionMLP_label(512, 300, 512, 768)
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=opt.multi_head)
        self.decoder_last = nn.TransformerDecoder(decoder_layer, num_layers=opt.decoder_num)
        self.decoder_images = nn.TransformerDecoder(decoder_layer, num_layers=opt.decoder_num)
        self.wordvec_proj = torch.nn.Linear(300, 768)
        self.textver_proj = torch.nn.Linear(512, 768)
        self.mlp = ProjectionMLP(768, 768)
        self.fc = torch.nn.Linear(768, 1)
        # self.awl = AutomaticWeightedLoss(2).cuda()
        self.bce = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()
        self.ce = torch.nn.CrossEntropyLoss(reduction='mean')
        self.asy = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)
        # self.transformer = nn.Transformer(768, 8, 6, 6)
        self.multihead_attn = nn.MultiheadAttention(768, 8, dropout=0.1)

    def forward(self, image, text_features, word_vec, labels, Train=False):
        bs = image.shape[0]
        # print('labels.shape',labels.shape)
        # image_features = self.img_encoder(image).last_hidden_state.transpose(0, 1)
        image_features = self.img_encoder(image).last_hidden_state.transpose(0, 1)
        text_features = torch.nn.functional.relu(self.textver_proj(text_features.cuda())).transpose(0, 1).expand(-1, bs,-1)
        decoder_images = self.decoder_images(image_features, text_features)  # image_features,text_features
        # print('text_features.shape',text_features.shape)
        # print('image_features.shape',image_features.shape)
        # image_features,_ = self.multihead_attn(image_features,text_features,text_features)
        # print("image_features.shape",image_features.shape)
        # if len(image_features.shape) == 4:  # [bs,2048, 7,7]
        #     image_features = image_features.flatten(2).transpose(1, 2)
        tgt = torch.nn.functional.relu(self.wordvec_proj(word_vec.float().cuda())).unsqueeze(1).expand(-1, bs,
                                                                                               -1)  # self.projection_label(text_features.transpose(0, 1), word_vec)
        h = self.decoder_last(tgt, decoder_images)
        h = h.transpose(0, 1)
        logits = self.fc(h).squeeze()
        if Train:
            loss = self.bce(logits, labels.squeeze().float())  # self.asy(logits, labels.squeeze().float())
y            return loss
        else:
            return logits


class CLIP_Decoder(nn.Module):
    def __init__(self, model_name="clip-vit-base-patch32",
                 freeze=True):  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        super(CLIP_Decoder, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels, Train=False):
        # inputs = self.processor(text=text, images=image, return_tensors="pt",padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        if Train:
            loss = self.bce(logits_per_image, labels)
            return loss
        else:
            return logits_per_image
