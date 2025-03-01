import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.eval import eval_net
from lib.FMASTransUNet import UNet

from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader, test_dataset


train_img_dir = '/zyy/DS-TransUNet-master/data/train/image/'
train_mask_dir = '/zyy/DS-TransUNet-master/data/train/mask/'
val_img_dir = '/zyy/DS-TransUNet-master/data/val/image/'
val_mask_dir = '/zyy/DS-TransUNet-master/data/val/mask/'
dir_checkpoint = '/zyy/DS-TransUNet-master/checkpoints'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cal(loader):
    tot = 0
    for batch in loader:
        imgs, _ = batch
        tot += imgs.shape[0]
    return tot


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def train_net(net,
              device,
              epochs=500,
              batch_size=4,
              lr=2e-4,
              save_cp=True,
              n_class=1,
              img_size=512,
              checkpoint_path='/zyy/DS-TransUNet-master/checkpoints/FMAS9x9.pth'):  # 自定义保存文件的路径

    train_loader = get_loader(train_img_dir, train_mask_dir, batchsize=batch_size, trainsize=img_size,
                              augmentation=False)
    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=1, trainsize=img_size, augmentation=False)

    n_train = cal(train_loader)
    n_val = cal(val_loader)
    logger = get_logger('FMAS9x9.log')

    logger.info(f'''开始训练:
        迭代次数:       {epochs}
        批大小:         {batch_size}
        学习率:         {lr}
        训练集大小:     {n_train}
        验证集大小:     {n_val}
        是否保存检查点: {save_cp}
        设备:           {device.type}
        图像大小:       {img_size}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 5, lr / 10)
    criterion = nn.CrossEntropyLoss() if n_class > 1 else nn.BCEWithLogitsLoss()

    best_dice = 0
    size_rates = [384, 512, 640]

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        best_cp = False
        Batch = len(train_loader)

        with tqdm(total=n_train * len(size_rates), desc=f'第 {epoch + 1}/{epochs} 轮', unit='img') as pbar:
            for batch in train_loader:
                for rate in size_rates:
                    imgs, true_masks = batch
                    trainsize = rate
                    if rate != 512:
                        imgs = F.upsample(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        true_masks = F.upsample(true_masks, size=(trainsize, trainsize), mode='bilinear',
                                                align_corners=True)

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_class == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)

                    masks_pred, l2, l3 = net(imgs)
                    loss1 = structure_loss(masks_pred, true_masks)
                    loss2 = structure_loss(l2, true_masks)
                    loss3 = structure_loss(l3, true_masks)
                    loss = 0.6 * loss1 + 0.2 * loss2 + 0.2 * loss3
                    
                    epoch_loss += loss.item()

                    pbar.set_postfix(**{'损失 (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])

        scheduler.step()

        # 验证阶段，计算各项指标
        val_metrics = eval_net(net, val_loader, device, n_class=n_class)
        val_dice = val_metrics['dice']
        val_iou = val_metrics['iou']
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']

        if val_dice > best_dice:
            best_dice = val_dice
            best_cp = True

        epoch_loss = epoch_loss / Batch
        logger.info(
            '第{}轮: 训练损失: {:.3f}, 验证Dice系数: {:.3f}, 验证IoU: {:.3f}, 验证Precision: {:.3f}, 验证Recall: {:.3f}, 验证F1: {:.3f}, 最佳Dice系数: {:.3f}'.format(
                epoch + 1, epoch_loss, val_dice * 100, val_iou * 100, val_precision * 100, val_recall * 100, val_f1 * 100, best_dice * 100))

        if save_cp and best_cp:
            torch.save(net.state_dict(), checkpoint_path)
            logging.info(f'检查点已保存至 {checkpoint_path}')


# 使用示例
# train_net(net, device, epochs=10, batch_size=1, lr=0.01, save_cp=True, n_class=1, img_size=512, checkpoint_path='custom_checkpoint.pth')


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=512,
                        help='The size of the images')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint.pth',
                        help='Path to save the checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(128, 1)
    net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)