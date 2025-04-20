import torch
import torch.nn as nn
import torch.nn.functional as TF


class FocalLoss(nn.Module):
    #alpha: Balances importance of different classes
    #       Default is 1.0 (no change
    #gamma: Controls the focus on harder examples. Larger values place more emphasis on hard-to-classify samples.
    #       Default is 2.0.(no change
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = TF.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()



class YOLOLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16, device=None):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.device = device
        self.focal = FocalLoss()

    def forward(self, raw_preds, raw_targets):
        total_loss = 0.0
        for key, pred in raw_preds.items():
            B, _, H, W = pred["bbox"].shape
            bbox_preds = pred["bbox"].view(B, 4, self.reg_max, H, W).permute(0, 3, 4, 1, 2)
            cls_pred = pred["cls"].permute(0, 2, 3, 1)

            targets = self.build_targets(raw_targets, H, W, self.device)
            cls_targets = targets["cls"]

            cls_loss = self.focal(cls_pred, cls_targets)
            total_loss += cls_loss

        return total_loss

    def build_targets(self, raw_targets, H, W, device):
        B = len(raw_targets)
        bbox_targets = torch.zeros(B, H, W, 4, device=device)
        cls_targets = torch.zeros(B, H, W, self.num_classes, device=device)

        for batch_idx, targets in enumerate(raw_targets):
            for i in range(0, len(targets), 5):  # Tag 5 værdier ad gangen
                target = targets[i:i+5]  # increment 5 værdier ad gangen
                
                c, x, y, w, h = target

                tile_i = int(x * W)
                tile_j = int(y * H)

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





                
        


        



