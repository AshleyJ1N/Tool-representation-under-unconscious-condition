"""
GPT2BPETokenizer text feature extract
"""

import os
import torch
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

choice = 'base'

output_path_sentence_feature = rf'F:\TE_DCNN_RSA\DCNNs\sementic_vs_image\GPT2'

# 加载GPT2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# 定义要提取特征的词
words = ["an axe", "a spanner", "a brush", "a toothbrush", "a fork",
          "a knife", "a wrech", "a spoon", "a tong", "a saw",
          "a turner", "a pruner", "a rolling pin", "a tweezer", "a whisk",
          "a screwdriver", "a plunger", "a pick up grabber", "a plier", "a hammer",
          "a sharpener", "an apple slicer", "a broom", "a comb", "a cleaver",
          "a pot brush", "a strawberry slicer", "a pizza cutter", "a washing sponge", "a bamboo dustpan",
          "a stapler", "a clean ball with a handle", "a shaver", "a bath ball", "a mango slicer",
          "a dustpan", "a table tennis racket", "a staple remover", "a carpet brusher", "a paper fan",
          "a belt", "a coca cola bottle", "a cigar", "a necktie", "a torch",
          "an umbrella", "a pearl necklace", "a lighter", "a bottle", "a facial cleanser",
          "a chocolate bar", "a cigarette", "a power strip", "a vacuum bottle", "a battery",
          "a watch", "a flashlight", "a laptop battery", "an eggplant", "a toothpaste",
          "a feather shuttle", "a bread", "a U-shape pillow", "a plate", "a chocolate packaging box",
          "a notebook", "a brick", "a kraft paper bag", "a button", "a Porcelain bottle",
          "a rubik's cube", "a adhesive tape", "a baby trainer cup", "a facial cream", "a hat",
          "a coin", "a tennis ball", "a clock", "a camera", "a tape"]

# 提取每个词的特征向量
for i in tqdm(range(len(words))):
    # 对词进行编码和输入到BERT模型
    input_ids = torch.tensor(tokenizer.encode(words[i])).unsqueeze(0)  # 分词处理
    outputs = model(input_ids)
    pooled_output = outputs[1]  # 句向量
    sentence_output = pooled_output.detach().numpy()
    # 将特征向量保存到本地文件
    filename = f'text_{i}_feats.mat'
    # print(sentence_output)
    savemat(os.path.join(output_path_sentence_feature, filename), {'sentence_output': sentence_output})

print('Word features saved.')
