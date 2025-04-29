import torch
import torch.nn as nn
import torch.nn.functional as TF

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16, device=None):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.device = device

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, raw_preds, raw_targets):
        total_cls_loss = 0.0
        total_bbox_loss = 0.0

        for key, pred in raw_preds.items():
            B, _, H, W = pred["bbox"].shape
            bbox_preds = pred["bbox"].view(B, 4, self.reg_max, H, W).mean(2).permute(0, 2, 3, 1)  # shape: [B, H, W, 4]
            cls_preds = pred["cls"].permute(0, 2, 3, 1)  # shape: [B, H, W, C]

            targets = self.build_targets(raw_targets, H, W, self.device)

            total_bbox_loss += self.l1_loss(bbox_preds, targets["bbox"])
            total_cls_loss += self.bce_loss(cls_preds, targets["cls"])

        return total_bbox_loss + total_cls_loss

    def build_targets(self, raw_targets, H, W, device):
        B = len(raw_targets)
        bbox_targets = torch.zeros(B, H, W, 4, device=device)
        cls_targets = torch.zeros(B, H, W, self.num_classes, device=device)

        for batch_idx, targets in enumerate(raw_targets):
            for target in targets:
                c, x, y, w, h = target.tolist()
                tile_i = min(int(x * W), W - 1)
                tile_j = min(int(y * H), H - 1)

                x_offset = (x * W) - (tile_i + 0.5)
                y_offset = (y * H) - (tile_j + 0.5)

                bbox_targets[batch_idx, tile_j, tile_i] = torch.tensor([x_offset, y_offset, w, h], device=device)
                cls_targets[batch_idx, tile_j, tile_i, int(c)] = 1.0

        return {
            "bbox": bbox_targets,
            "cls": cls_targets
        }

    def distributed_focal_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        tl = target.long
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (TF.cross_entropy(pred_dist, tl.view(-1), reduction ='none').view(tl.shape) * wl + TF.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)





                
        


        



