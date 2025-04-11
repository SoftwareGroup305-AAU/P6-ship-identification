import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16, device=None):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.device = device
    
    def forward(self, raw_preds, raw_targets):
       for key, pred in raw_preds.items():
           B, _, H, W = pred["bbox"].shape
           bbox_preds = pred["bbox"].view(B, 4, self.reg_max, H, W).permute(0, 3, 4, 1, 2)
           cls_pred = pred["cls"].permute(0, 2, 3, 1)
           self.build_targets(raw_targets, H, W, self.device)

    def build_targets(self, raw_targets, H, W, device):
        B = len(raw_targets)
        bbox_targets = torch.zeros(B, H, W, 4, device=device)
        cls_targets = torch.zeros(B, H, W, self.num_classes, device=device)
        for batch_idx, targets in enumerate(raw_targets):
            for target in targets:
                c, x, y, w, h = target

                tile_i = int(x*W)
                tile_j = int(y*H)

                x_offset = (x * W) - (tile_i + 0.5)  
                y_offset = (y * H) - (tile_j + 0.5)

                bbox_targets[batch_idx, tile_j, tile_i] = torch.tensor([x_offset, y_offset, w, h], device=device)
                cls_targets[batch_idx, tile_j, tile_i, int(c)] = 1.0
        return {
            "bbox": bbox_targets,
            "cls": cls_targets
        }















                
        


        



