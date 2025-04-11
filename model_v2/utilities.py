import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
    
    def forward(self, preds, raw_targets):
       for pred in preds.values():
           B, _, H, W = pred["bbox"].shape
           bbox_preds = pred["bbox"].view(B, 4, self.reg_max, H, W).permute(0, 3, 4, 1, 2)
           cls_pred = pred["cls"]



