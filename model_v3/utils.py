import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import ops
import math
from dataset import YOLODataset
from core import YOLO
from utils import build_dfl_targets
import torch.nn.functional as F

def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2, = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)

def CompLoss():
    
    return (CIoULoss() * 0.6) + (distributed_focal_loss() * 0.4)

def CIoULoss(pred, target: torch.Tensor):
    obj_mask = (target > 0)
    
    loss = ops.complete_box_iou_loss()
    return loss

def distributed_focal_loss(pred, target, reg_max=16):
    """
    pred: model output dict with 'p3' containing 'bbox' logits of shape [B, 4 * reg_max, H, W]
    target: float tensor of shape [B, H, W, 4], continuous values in [0, reg_max)
    """
    bbox = pred['p3']['bbox']

    B, C, H, W = bbox.shape
    assert C == 4 * reg_max, f"Expected {4 * reg_max} channels, got {C}"

    # Reshape to proper format
    bbox = bbox.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2).contiguous()

    target = target.to(bbox.device).float()

    total_loss = 0.0
    for i in range(4):  # For each of the 4 box coordinates
        pred_dist = bbox[..., i, :]
        t = target[..., i]            

        tl = t.long()
        tr = tl + 1
        wl = tr - t
        wr = 1 - wl

        tl = torch.clamp(tl, 0, reg_max - 1)
        tr = torch.clamp(tr, 0, reg_max - 1)

        loss_l = F.cross_entropy(pred_dist.view(-1, reg_max), tl.view(-1), reduction='none').view(t.shape)
        loss_r = F.cross_entropy(pred_dist.view(-1, reg_max), tr.view(-1), reduction='none').view(t.shape)
        dfl = loss_l * wl + loss_r * wr

        total_loss += dfl.mean()

    return total_loss / 2  # average across 4 box coordinates