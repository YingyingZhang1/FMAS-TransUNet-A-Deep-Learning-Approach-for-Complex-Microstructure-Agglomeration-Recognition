import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.dataloader import get_loader, test_dataset  # 导入数据加载函数
from lib.FMASTransUNet import UNet  # 导入模型定义


# 确保使用与训练时相同的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型权重
checkpoint_path = '/zyy/DS-TransUNet-master/checkpoints/checkpoint.pth'  # 替换为实际的模型权重文件路径
net = UNet(128, 1)  # 根据您的模型定义进行调整
net = nn.DataParallel(net)
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net.eval()  # 设置模型为评估模式
net.to(device)

# 假设您已经有了一个用于加载测试数据的函数，类似于之前的get_loader但用于测试集
# 如果没有，请按照之前的逻辑创建一个
def get_test_loader(img_dir, mask_dir, batch_size=1, trainsize=512):
    # 此处应实现加载测试数据集的逻辑
    pass

# 初始化测试数据加载器
test_img_dir = '/zyy/DS-TransUNet-master/data/val/image'  # 测试图像目录
test_mask_dir = '/zyy/DS-TransUNet-master/data/val/mask'  # 测试标签目录
test_loader = get_test_loader(test_img_dir, test_mask_dir, batch_size=1, trainsize=512)

# 可视化预测结果的函数
def visualize_results(image, mask, pred_mask, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.permute(1, 2, 0))  # 假设image是RGB图像
    axs[0].set_title('Image')
    axs[1].imshow(mask.squeeze(), cmap='gray')  # 假设mask是单通道的
    axs[1].set_title('Ground Truth Mask')
    axs[2].imshow(pred_mask.squeeze(), cmap='gray')
    axs[2].set_title('Predicted Mask')
    plt.suptitle(title)
    plt.show()

# 对测试集中的每个样本进行预测并可视化
with torch.no_grad():  # 在评估时不计算梯度
    for i, batch in enumerate(test_loader):
        image, true_mask = batch
        image, true_mask = image.to(device), true_mask.to(device)
        
        # 进行预测
        pred_mask = net(image)
        pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()  # 将概率转换为二值掩码
        
        # 将Tensor转换为CPU上的numpy数组以便于可视化
        image_np = image.cpu().numpy().transpose((1, 2, 0))  # 调整维度顺序以适应matplotlib
        true_mask_np = true_mask.cpu().numpy()
        pred_mask_np = pred_mask.cpu().numpy()
        
        # 可视化
        visualize_results(image_np, true_mask_np, pred_mask_np, f'Sample {i+1}')
        
        # 如果只需要查看几个样本，可以在这里添加break
        if i >= 4:  # 例如，只查看前5个样本
             break

print("可视化完成。")
