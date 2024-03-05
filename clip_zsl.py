import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import clip
from PIL import Image
#
# device = 'cpu'#"cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# model, preprocess = clip.load("ViT-B/32", device=device)#ViT-B/32
# mod = clip.available_models()
# print(model)
# for parameter in model.visual.parameters():
#     parameter.requires_grad = False
#
# # self.clip_model.transformer.
# # print(model.visual.transformer.resblocks)
# # for parameter in model.visual.ResidualAttentionBlock():
# #     print(parameter)
# #     parameter.requires_grad = True
# img_raw = Image.open("dog.png")
# image = preprocess(Image.open("dog.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size
# vision = model.visual
# vision_img_output = vision(image)
# # text = model.transformer
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     logits_per_image_b = image_features @ text_features.T
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# #
# # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# import torch
# import torch.nn as nn
# #
# decode_layer = nn.TransformerDecoderLayer(d_model=512,nhead=8)  # d_model is the input feature, nhead is the number of head in the multiheadattention
# memory = torch.ones(1,32,512)  # the sequence from the last layer of the encoder ; 可以类比为: batch_size * seqence_length * hidden_size
# tgt = torch.zeros(20,32,512)  # the sequence to the decoder layer
# out = decode_layer(tgt, memory)
# print(out.shape)  # 20*20*512
# from PIL import Image
# import requests
# from transformers import CLIPProcessor, CLIPVisionModel,CLIPModel
#
# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs = processor(images=image, return_tensors="pt")
#
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# pooled_output = outputs.pooler_output  # pooled CLS states
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
