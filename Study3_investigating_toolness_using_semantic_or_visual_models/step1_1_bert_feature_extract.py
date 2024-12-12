"""
BERT text feature extract
"""

import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

choice = 'base'

# output_path_sentence_feature = rf'F:\TE_DCNN_RSA\DCNNs\sementic_vs_image\BERT\{choice}\sentence_feature'
path = 'F:\TE_DCNN_RSA\DCNNs\stimuli'

# 加载BERT模型和分词器
if choice == 'large':
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
elif choice == 'base':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, )

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

# print(model)
# 提取每个词的特征向量
with torch.no_grad():
    inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    all_hidden_states = outputs.hidden_states
    for idx, hidden_state in enumerate(all_hidden_states):
        output_path = rf'F:\TE_DCNN_RSA\DCNNs\sementic_vs_image\BERT\python\base\each_layer\layer{idx}'
        hidden_state=hidden_state.detach().numpy()
        print(hidden_state[0].shape)
        for i in range(0,80):
            filename = f'text_{i + 1}_feats.mat'
            savemat(os.path.join(output_path, filename), {f'layer{idx}': hidden_state[i][-1]})


print('Word features saved.')

