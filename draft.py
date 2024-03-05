import torch
import numpy as np
def _get_classes(file_tag1k,file_tag81):
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
# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)
# print(transformer_model)
# _,_,nus_class_name_925 = _get_classes('datasets/NUS-WIDE/NUS_WID_Tags/TagList1k.txt','datasets/NUS-WIDE/ConceptsList/Concepts81.txt')
# labels = torch.ones(32,925)
# nus_class_name_925 = np.array(nus_class_name_925)
# t = torch.gt(labels,0)
# print(nus_class_name_925)
# label_classname = []
# for i in range(labels.shape[0]):
#     label_classname.append(nus_class_name_925[t[i]])
# for i in all_label
# print('c1',all_label)
['a photo of the portugal.', 'a photo of the quebec.']
#
# img = cv2.imread("dog.png")
# width = img.shape[1]
# height = img.shape[2]
# img = cv2.resize(img, (250, 250))
#
# tran_tensor = transforms.ToTensor()
# img = tran_tensor(img)
# print("img.shape: ", img.shape)
# print("type(img): ", type(img))
#
# img = img.view(1, 3, 250, 250)
#
# writer = SummaryWriter("logs")
#
# Conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
#
# ConvTrans = nn.ConvTranspose2d(in_channels=512, out_channels=512,kernel_size =1)
#
# Maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
# MaxUnpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
#
# img = img.reshape(3, 250, 250)
# writer.add_image("input", img, 0)

# Transposed convolution
# img = img.reshape(1, 512, 1, 1)
# output_ConvTrans = ConvTrans(img)
# print("output_ConvTrans.shape: ", output_ConvTrans.shape)
# output_ConvTrans = output_ConvTrans.reshape(3, 300, 300)
# writer.add_image("output_ConvTrans", output_ConvTrans, 1)
# train_x = [torch.tensor([1, 2, 3, 4, 5, 6, 7]),
#            torch.tensor([2, 3, 4, 5, 6, 7]),
#            torch.tensor([3, 4, 5, 6, 7]),
#            torch.tensor([4, 5, 6, 7]),
#            torch.tensor([5, 6, 7]),
#            torch.tensor([6, 7]),
#            torch.tensor([7])]
# d = torch.nn.utils.rnn.pad_sequence(train_x)
# print(d)
# multihead_attn = nn.MultiheadAttention(512, 8)
# a = torch.rand([925,24,512])
# b = torch.rand([20,24,512])
# mask = torch.zeros([925,20])
# mask = torch.gt(mask,0).cpu()
# attn_mask=torch.randint(0,2,[3,3]).byte()
# attn_output, attn_output_weights = multihead_attn(a,b,b,attn_mask =mask)
# print(attn_output)
# #  F.interpolate
# out_interpolate = F.interpolate(img, scale_factor=2, mode='bilinear')
# print("out_interpolate.shape: ", out_interpolate.shape)
# out_interpolate = out_interpolate.reshape(3, 500, 500)
# writer.add_image("out_interpolate: ", out_interpolate, 2)
# a = torch.rand([925,24,512])
# decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=4)
# decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=2)
# decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=2)
# a = torch.rand([1,1024,158])
# b = torch.rand([20,24,512])
# c = torch.rand([925,24,512])
# transformers = nn.Transformer(158, 2, 6, 6)
# out = transformers(a)
# import torch.nn.functional as F
# a = torch.rand([1024,1,158])
# c = F.normalize(a)
# print(c)
# print(out)
# decoder_out1 = decoder_1(a,b)
# decoder_out2 = decoder_2(c,decoder_out1)
# output = decoder_out1
# Unpooling
# out_maxpool, indices = Maxpool(img)
# print("out_maxpool.shape: ", out_maxpool.shape)
# out_maxpool_1 = out_maxpool.reshape(3, 125, 125)
# writer.add_image("out_maxpool: ", out_maxpool_1, 3)
# out_maxunpool = MaxUnpool(out_maxpool, indices)
# print("out_maxunpool.shape: ", out_maxunpool.shape)
# out_maxunpool = out_maxunpool.reshape(3, 250, 250)
# writer.add_image("out_maxunpool: ", out_maxunpool, 4)
# writer.close()
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#
#     def forward(self, Q, K, V, attn_mask):
#         '''
#         Q: [batch_size, n_heads, len_q, d_k]
#         K: [batch_size, n_heads, len_k, d_k]
#         V: [batch_size, n_heads, len_v(=len_k), d_v]
#         attn_mask: [batch_size, n_heads, seq_len, seq_len]
#         '''
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
#         scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
#
#         attn = nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
#         return context, attn
# class MultiHeadAttention(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention, self).__init__()
#         self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
#         self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
#         self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
#         self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
#     def forward(self, input_Q, input_K, input_V, attn_mask):
#         '''
#         input_Q: [batch_size, len_q, d_model]
#         input_K: [batch_size, len_k, d_model]
#         input_V: [batch_size, len_v(=len_k), d_model]
#         attn_mask: [batch_size, seq_len, seq_len]
#         '''
#         residual, batch_size = input_Q, input_Q.size(0)
#         # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
#         K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
#         V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
#
#         attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
#
#         # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
#         context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
#         context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
#         output = self.fc(context) # [batch_size, len_q, d_model]
#         return nn.LayerNorm(d_model).cuda()(output + residual), attn
# from sklearn.metrics import average_precision_score
# def compute_mAP(labels,outputs):
#     y_true = labels
#     y_pred = outputs
#     AP = []
#     classnum = y_true.shape[1]
#     for i in range(y_true.shape[1]):
#         # AP.append(average_precision_score(y_true[:,i],y_pred[:,i]))
#         AP.append(average_precision_score(y_true[:,i], y_pred[:,i]))
#     return np.mean(AP)
# # precision_score,f1_score,recall_score
# classes = ['green', 'black', 'red', 'blue']
# targetSrc = [[0,1,1,1], [0,0,1,0], [1,0,0,1], [1,1,1,0], [1,0,0,0]]
# predSrc = [[0,1,0,1], [0,0,1,1], [1,0,0,1], [1,0,1,0], [1,0,0,0]]
# # map是通过sigmoid计算的
# target = np.array(targetSrc)
# pred = np.array(predSrc)
# map = compute_mAP(target,pred)
# print(map)
# import numpy as np
# import pickle
# import os
#
# from sklearn.metrics import hamming_loss
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import jaccard_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import coverage_error
# from sklearn.metrics import label_ranking_average_precision_score
# from sklearn.metrics import label_ranking_loss
#
# def get_top_k_results(hat_y, y, k=1):
#     format_hat_y = np.zeros(np.shape(hat_y))
#     for i in range(len(hat_y)):
#         i_s = np.argsort(hat_y[i, :])[-k:]
#         for j in i_s:
#             format_hat_y[i, j] = 1.
#
#     values = {}
#     values['hamming_loss'+'@'+str(k)] = hamming_loss(y, format_hat_y)
# #    values['jaccard_score_micro'+'@'+str(k)] = jaccard_score(y, format_hat_y, average = 'micro')
# #    values['jaccard_score_macro'+'@'+str(k)] = jaccard_score(y, format_hat_y, average = 'macro')
#     values['recall'+'@'+str(k)] = np.mean(np.sum(format_hat_y * y, axis=1)/ np.sum(y, axis=1))
#     values['precision'+'@'+str(k)] = np.mean(np.sum(format_hat_y * y, axis=1)/ np.sum(format_hat_y, axis=1))
#     return values
#
# def get_avg_results(hat_y, y):
#     values = {}
#     values['avg_precision_micro'] = average_precision_score(y, hat_y, average = 'micro')
# #    values['avg_precision_macro'] = average_precision_score(y, hat_y, average = 'macro')
#     values['roc_auc_score_micro'] = roc_auc_score(y, hat_y, average = 'micro')
# #    values['roc_auc_score_macro'] = roc_auc_score(y, hat_y, average = 'macro')
#     values['coverage_error'] = coverage_error(y, hat_y)
#     values['label_ranking_average_precision_score'] = label_ranking_average_precision_score(y, hat_y)
#     values['label_ranking_loss'] = label_ranking_loss(y, hat_y)
#     return values
#
# def evaluator(hat_y, y):
#     values = get_avg_results(hat_y, y)
#     values_2 = get_top_k_results(hat_y, y, k=1)
#     values_3 = get_top_k_results(hat_y, y, k=3)
#     values.update(values_2)
#     values.update(values_3)
#     return values
#
# if __name__ == '__main__':
#     y = np.array([[0., 1., 0.],[1., 0., 1.]])
#     hat_y = np.array([[0.3, 0.7, 0.1],[0.1, 0.2, 0.8]])
#     c = np.rint(hat_y)
#     z = evaluator(hat_y, y)
#     print (z)
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state