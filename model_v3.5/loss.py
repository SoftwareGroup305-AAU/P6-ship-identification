import torch
import torch.nn as nn
import torch.nn.functional as TF
from utils import bbox2dist   
# class YOLOLoss(nn.Module):
#     def __init__(self, num_classes, reg_max=16, device=None):
#         super().__init__()
#         self.reg_max = reg_max
#         self.num_classes = num_classes
#         self.device = device
    
#     def forward(self, raw_preds, raw_targets):
#        for key, pred in raw_preds.items():
#            B, _, H, W = pred["bbox"].shape
#            bbox_preds = pred["bbox"].view(B, 4, self.reg_max, H, W).permute(0, 3, 4, 1, 2)
#            cls_pred = pred["cls"].permute(0, 2, 3, 1)
           
#            targets = self.build_targets(raw_targets, H, W, self.device)

#     def build_targets(self, raw_targets, H, W, device):
#         B = len(raw_targets)
#         bbox_targets = torch.zeros(B, H, W, 4, device=device)
#         cls_targets = torch.zeros(B, H, W, self.num_classes, device=device)
#         for batch_idx, targets in enumerate(raw_targets):
#             for target in targets:
#                 c, x, y, w, h = target

#                 tile_i = int(x*W)
#                 tile_j = int(y*H)

#                 x_offset = (x * W) - (tile_i + 0.5)  
#                 y_offset = (y * H) - (tile_j + 0.5)

#                 bbox_targets[batch_idx, tile_j, tile_i] = torch.tensor([x_offset, y_offset, w, h], device=device)
#                 cls_targets[batch_idx, tile_j, tile_i, int(c)] = 1.0
#         return {
#             "bbox": bbox_targets,
#             "cls": cls_targets
#         }

#     def distributed_focal_loss(pred_dist, target):
#         """Return sum of left and right DFL losses."""
#         tl = target.long
#         tr = tl + 1
#         wl = tr - target
#         wr = 1 - wl
#         return (TF.cross_entropy(pred_dist, tl.view(-1), reduction ='none').view(tl.shape) * wl + TF.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

#     nn.bbox2
class YOLOLoss:
    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.args
        #self.reg_max = m.reg_max
        #self.device = device

        self.bbox_loss = BboxLoss(m.reg_max - 1).to(device)
        self.proj = torch.arrange(m.reg_max, dtype=torch.float, device=device)

class BboxLoss(nn.Module):
    def __init__(self, reg_max):
        super.__init__()
        self.reg_max = reg_max
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_boxes, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IOU loss"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        target_bbox = bbox2dist(anchor_points, target_bboxes, self.reg_max)
        loss_dfl = self.distributed_focal_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_bbox[fg_mask]) * weight
        loss = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    def distributed_focal_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        tl = target.long
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (TF.cross_entropy(pred_dist, tl.view(-1), reduction ='none').view(tl.shape) * wl + TF.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
                
        


        



