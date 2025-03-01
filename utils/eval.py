import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def dice_coeff(pred, gt, smooth=1e-5):
    intersection = (pred * gt).sum()
    unionset = pred.sum() + gt.sum()
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice

def iou_coeff(pred, gt, smooth=1e-5):
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def precision_and_recall(pred, gt, smooth=1e-5):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall

def f1_score(precision, recall, smooth=1e-5):
    return (2 * precision * recall + smooth) / (precision + recall + smooth)

def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    tot_dice = 0
    tot_iou = 0
    tot_precision = 0
    tot_recall = 0
    tot_f1 = 0
    n_val = len(loader)
    N = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            #mask_pred = net(imgs)
            mask_pred, _, _ = net(imgs)
            if n_class > 1:
                loss = F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                dice = dice_coeff(pred, true_masks)
                iou = iou_coeff(pred, true_masks)
                precision, recall = precision_and_recall(pred, true_masks)
                f1 = f1_score(precision, recall)
                tot_dice += dice.item()
                tot_iou += iou.item()
                tot_precision += precision.item()
                tot_recall += recall.item()
                tot_f1 += f1.item()
                N += imgs.size(0)
            pbar.update()

    return {
        'dice': tot_dice / N,
        'iou': tot_iou / N,
        'precision': tot_precision / N,
        'recall': tot_recall / N,
        'f1': tot_f1 / N,
    }
