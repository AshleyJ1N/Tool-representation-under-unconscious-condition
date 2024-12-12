
"""
    使用去除了所有工具标签的ImageNet2012（LSVRC2012，共925分类）训练的alexnet
    训练参数：
    batch size = 128, momentum = 0.9, weight decay = 0.0005, training epoch = 90,
    learning rate = 0.01, learning rate decay = 10 times for every 30 epochs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import ImageFile
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

ImageFile.LOAD_TRUNCATED_IMAGES = True

seed = 1
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 定义超参数
BATCH_SIZE = 128  # 每批处理的数据
DEVICE = torch.device('cuda:0')  # 放在cuda上训练
EPOCHS = 90  # 训练数据集的轮次
modellr = 0.01
beta1 = 0.9
beta2 = 0.999
weightdecay = 0.0005
MOMENTUM = 0.9

# 构建pipeline，对图像做处理
pipeline = transforms.Compose([
    # 分辨率重置为256
    transforms.Resize(256),
    # 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像(因为这图片像素不一致直接统一)
    transforms.CenterCrop(224),
    # 将图片转成tensor
    transforms.ToTensor(),
    # 正则化，模型出现过拟合现象时，降低模型复杂度
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图片路径(训练图片和测试图片的)
base_dir_train = r'train'
base_dir_val = r'val'

# 加载数据集
train_dataset = datasets.ImageFolder(root=base_dir_train, transform=pipeline)
print("train_dataset=" + repr(train_dataset[1][0].size()))
print("train_dataset.class_to_idx=" + repr(train_dataset.class_to_idx))
# 创建训练集的可迭代对象，一个batch_size地读取数据,shuffle设为True表示随机打乱顺序读取
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 测试集
val_dataset = datasets.ImageFolder(root=base_dir_val, transform=pipeline)
print(val_dataset)
print("val_dataset=" + repr(val_dataset[1][0].size()))
print("val_dataset.class_to_idx=" + repr(val_dataset.class_to_idx))
# 创建测试集的可迭代对象，一个batch_size地读取数据
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 获得一批测试集的数据
images, labels = next(iter(val_loader))
print(images.shape)
print(labels.shape)

# 损失函数,交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用未经预训练的模型
model = torchvision.models.alexnet(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 1000)
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=modellr, weight_decay=weightdecay, momentum=MOMENTUM)
# 定义学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch衰减0.1

train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []
train_iteration_list = []
test_iteration_list = []


# 定义训练方法
def train(model, device, train_loader, optimizer, epoch, scheduler):
    train_correct = 0.0
    model.train()
    sum_loss = 0.0
    iteration = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, label) in enumerate(train_loader):
        # 获取数据与标签
        data, label = Variable(data).to(device), Variable(label).to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 计算损失
        output = model(data)
        loss = criterion(output, label)

        # 反向传播
        loss.backward()

        print_loss = loss.data.item()
        sum_loss += print_loss
        _, train_predict = torch.max(output.data, 1)

        if torch.cuda.is_available():
            train_correct += (train_predict.cuda() == label.cuda()).sum()
        else:
            train_correct += (train_predict == label).sum()
        accuracy = (train_correct / total_num) * 100
        print("Epoch: %d , Batch: %3d , Loss : %.8f,train_correct:%d , train_total:%d , accuracy:%.6f" % (
            epoch + 1, batch_idx + 1, loss.item(), train_correct, total_num, accuracy))
        # 存在集合画图
        if (epoch + 1) == EPOCHS:  # 只画出最后一个epoch时候的准确度变化曲线
            iteration += 1
            train_loss_list.append(loss.item())
            train_iteration_list.append(iteration)
            train_accuracy_list.append(accuracy)


# 定义验证方法
def val(model, device, val_loader, epoch):
    print("=====================预测开始=================================")
    iteration = 0
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total_num = len(val_loader.dataset)
    print(total_num, len(val_loader))
    with torch.no_grad():
        for data, target in val_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            if torch.cuda.is_available():
                correct += torch.sum(pred.cuda() == target.cuda())
            else:
                correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        acc = correct / total_num * 100
        avg_loss = test_loss / len(val_loader)
        """
            每次结束训练一次之后调用
        """
        iteration += 1
        # 存入集合准备画图
        test_loss_list.append(avg_loss)
        test_accuracy_list.append(acc)
        test_iteration_list.append(epoch)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            avg_loss, correct, len(val_loader.dataset), acc))

        global best_val_acc
        # if acc > best_val_acc:
        #     best_val_acc = acc
        #     print("Best Accuracy:{:.6f}%".format(best_val_acc))
        #     torch.save(model.state_dict(), 'best-{:.6f}.model.pth'.format(best_val_acc))  # 保存模型


# 训练
for epoch in range(EPOCHS):
    train(model, DEVICE, train_loader, optimizer, epoch, scheduler)
    val(model, DEVICE, val_loader, epoch)
    # 更新学习率
    optimizer.step()

torch.save(model, 'alexnet_toolexcluded_90.pth')  # 保存模型

# 可视化测试机的loss和accuracy
plt.figure(1)
plt.plot(test_iteration_list, test_loss_list)
plt.title("test loss")
plt.ylabel("loss")
plt.xlabel("Number of test iteration")
plt.show()

plt.figure(2)
plt.plot(test_iteration_list, test_accuracy_list)
plt.title("test accuracy")
plt.xlabel("Number of test iteration")
plt.ylabel("accuracy")
plt.show()

# 可视化训练集loss和accuracy
plt.figure(3)
plt.plot(train_iteration_list, train_loss_list)
plt.title("train loss")
plt.xlabel("Number of train iteration")
plt.ylabel("accuracy")
plt.show()

plt.figure(4)
plt.plot(train_iteration_list, train_accuracy_list)
plt.title("train accuracy")
plt.xlabel("Number of train iteration")
plt.ylabel("accuracy")
plt.show()
