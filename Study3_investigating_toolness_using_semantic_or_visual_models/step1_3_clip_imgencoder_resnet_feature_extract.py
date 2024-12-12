"""
RN50 feature extract
"""

import torch
import clip
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

path = 'F:\TE_DCNN_RSA\DCNNs\stimuli'
output_path_feature = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN101\image_feature'
output_path_text_feature = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN101\text_feature'
output_path_image_text_feature = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN101\image_logits'

device = "cpu"
seed = 1
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
model, preprocess = clip.load("RN50", device=device)
model.eval()

text = clip.tokenize(["an axe", "a spanner", "a brush", "a toothbrush", "a fork",
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
                      "a coin", "a tennis ball", "a clock", "a camera", "a tape"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)  # text的feature
    for i in tqdm(range(1, 81)):
        os.chdir(path)
        pic = f'SHINEd_{i}_300.png'
        # 提取image的feature
        image = preprocess(Image.open(pic)).unsqueeze(0).to(device)
        image_features = model.encode_image(image) # image的feature
        filename1 = f'image_{i}_feats.npy'
        np.save(os.path.join(output_path_feature, filename1), image_features)

# 提取text的feature
for i in range(1, 81):
    filename2 = f'text_{i}_feats.npy'
    np.save(os.path.join(output_path_text_feature, filename2), text_features[i-1])
