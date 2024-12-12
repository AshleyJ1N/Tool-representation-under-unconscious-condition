import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

seed = 1
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

path = 'F:\TE_DCNN_RSA\DCNNs\stimuli'
output_path1 = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN50\extract_from_diff_layer\maxpool'
output_path2 = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN50\extract_from_diff_layer\layer1'
output_path3 = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN50\extract_from_diff_layer\layer2'
output_path4 = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN50\extract_from_diff_layer\layer3'
output_path5 = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN50\extract_from_diff_layer\avgpool'
output_path6 = r'F:\TE_DCNN_RSA\DCNNs\CLIP\RN50\extract_from_diff_layer\attnpool'

# 加载模型
resnet50 = models.resnet50(pretrained=True)
for name, layer in resnet50.named_children():
    print(name, layer)
resnet50.eval()  # 设置为评估模式，不进行反向传播

# 钩子函数
def hook_maxpooling(module, input, output):
    output = output.cpu().numpy()  # 将tensor转为numpy数组
    feats1.append(output)

def hook_layer1(module, input, output):
    output = output.cpu().numpy()  # 将tensor转为numpy数组
    feats2.append(output)


def hook_layer2(module, input, output):
    output = output.cpu().numpy()  # 将tensor转为numpy数组
    feats3.append(output)


def hook_layer3(module, input, output):
    output = output.cpu().numpy()  # 将tensor转为numpy数组
    feats4.append(output)

def hook_avgpool(module, input, output):
    output = output.cpu().numpy()  # 将tensor转为numpy数组
    feats5.append(output)

def hook_fc(module, input, output):
    output = output.cpu().numpy()  # 将tensor转为numpy数组
    feats6.append(output)

#   'max_pooling2d_1' (5), 'activation_10_relu' (37), 'activation_22_relu' (79),
#   'activation_40_relu' (141), 'avg_pool' (174), 'fc1000' (175)
handle1 = resnet50.maxpool.register_forward_hook(hook_maxpooling)
handle2 = resnet50.layer1[2].relu.register_forward_hook(hook_layer1)
handle3 = resnet50.layer2[3].relu.register_forward_hook(hook_layer2)
handle4 = resnet50.layer3[5].relu.register_forward_hook(hook_layer3)
handle5 = resnet50.avgpool.register_forward_hook(hook_avgpool)
handle6 = resnet50.fc.register_forward_hook(hook_fc)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

features1 = []
features2 = []
features3 = []
features4 = []
features5 = []
features6 = []
with torch.no_grad():
    for i in tqdm(range(1, 81)):
        os.chdir(path)
        # 加载图片
        pic = f'SHINEd_{i}_300.png'
        img = Image.open(pic).convert("RGB")
        # 对图片进行预处理并扩展一维作为模型的输入
        img_tensor = preprocess(img).unsqueeze(0)
        feats1 = []
        feats2 = []
        feats3 = []
        feats4 = []
        feats5 = []
        feats6 = []
        resnet50(img_tensor)
        features1.append(feats1[0])
        features2.append(feats2[0])
        features3.append(feats3[0])
        features4.append(feats4[0])
        features5.append(feats5[0])
        features6.append(feats6[0])
    features1 = np.concatenate(features1)
    features2 = np.concatenate(features2)
    features3 = np.concatenate(features3)
    features4 = np.concatenate(features4)
    features5 = np.concatenate(features5)
    features6 = np.concatenate(features6)


# 保存特征
for i in range(0, 80):
    filename1 = f'maxpool_{i + 1}_feats.npy'
    filename2 = f'layer1_{i + 1}_feats.npy'
    filename3 = f'layer2_{i + 1}_feats.npy'
    filename4 = f'layer3_{i + 1}_feats.npy'
    filename5 = f'avgpool_{i + 1}_feats.npy'
    filename6 = f'fc_{i + 1}_feats.npy'
    np.save(os.path.join(output_path1, filename1), features1[i])
    np.save(os.path.join(output_path2, filename2), features2[i])
    np.save(os.path.join(output_path3, filename3), features3[i])
    np.save(os.path.join(output_path4, filename4), features4[i])
    np.save(os.path.join(output_path5, filename5), features5[i])
    np.save(os.path.join(output_path6, filename6), features6[i])

handle1.remove()
handle2.remove()
handle3.remove()
handle4.remove()
handle5.remove()
handle6.remove()
