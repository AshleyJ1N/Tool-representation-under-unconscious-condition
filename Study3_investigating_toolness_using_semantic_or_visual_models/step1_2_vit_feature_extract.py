"""
ViT feature extract
"""

import torchvision.transforms as transforms
from transformers import ViTModel
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from scipy.io import savemat

model_choose = "vit16"
path = 'F:\TE_DCNN_RSA\DCNNs\stimuli'
output_path_feature = rf'F:\TE_DCNN_RSA\DCNNs\sementic_vs_image\ViT\{model_choose}\python\large'
device = "cpu"

if model_choose == "vit16":
    vit = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
elif model_choose == "vit32":
    vit = ViTModel.from_pretrained('google/vit-large-patch32-224-in21k')

# print(vit)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

features = []
with torch.no_grad():
    for i in tqdm(range(1, 81)):
        os.chdir(path)
        # 加载图片
        pic = f'SHINEd_{i}_300.png'
        img = Image.open(pic).convert("RGB")
        # 对图片进行预处理并扩展一维作为模型的输入
        img_tensor = preprocess(img).unsqueeze(0)
        feats = []
        output = vit(img_tensor)
        last_hidden_state = output.last_hidden_state.detach().numpy()
        pooler_output = output.pooler_output.detach().numpy()
        filename = f'output_{i}_feats.mat'
        savemat(os.path.join(output_path_feature, filename), {'last_hidder_state': last_hidden_state, 'pooler_output': pooler_output})


