import torch
import torch.nn as nn
import math
    
class GIoULoss(nn.Module):
    def forward(self, prediction, target):
        x1 = torch.min(prediction[:, 0], target[:, 0])
        y1 = torch.min(prediction[:, 1], target[:, 1])
        x2 = torch.max(prediction[:, 2], target[:, 2])
        y2 = torch.max(prediction[:, 3], target[:, 3])

        intersection_width = (x2 - x1).clamp(0)
        intersection_height = (y2 - y1).clamp(0)
        intersection = intersection_width * intersection_height

        predicted_area = (prediction[:, 2] - prediction[:, 0]) * (prediction[:, 3] - prediction[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = predicted_area + target_area - intersection

        iou = intersection / (union + 1e-6)

        enclosing_x1 = torch.min(prediction[:, 0], target[:, 0])
        enclosing_y1 = torch.min(prediction[:, 1], target[:, 1])
        enclosing_x2 = torch.max(prediction[:, 2], target[:, 2])
        enclosing_y2 = torch.max(prediction[:, 3], target[:, 3])

        enclosing_width = (enclosing_x2 - enclosing_x1).clamp(0)
        enclosing_height = (enclosing_y2 - enclosing_y1).clamp(0)
        c = enclosing_width * enclosing_height

        giou = iou - (c - union) / (c + 1e-6)
        loss = 1 - giou
        loss = loss.clamp(min=0)
        return loss.mean()
    
class CIoULoss(nn.Module):
    def forward(self, prediction, target):
        pred_cx = (prediction[:, 0] + prediction[:, 2]) / 2
        pred_cy = (prediction[:, 1] + prediction[:, 3]) / 2
        pred_w = (prediction[:, 2] - prediction[:, 0]).clamp(min=1e-7)
        pred_h = (prediction[:, 3] - prediction[:, 1]).clamp(min=1e-7)

        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        target_w = (target[:, 2] - target[:, 0]).clamp(min=1e-7)
        target_h = (target[:, 3] - target[:, 1]).clamp(min=1e-7)

        inter_x1 = torch.max(prediction[:, 0], target[:, 0])
        inter_y1 = torch.max(prediction[:, 1], target[:, 1])
        inter_x2 = torch.min(prediction[:, 2], target[:, 2])
        inter_y2 = torch.min(prediction[:, 3], target[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)

        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        enc_x1 = torch.min(prediction[:, 0], target[:, 0])
        enc_y1 = torch.min(prediction[:, 1], target[:, 1])
        enc_x2 = torch.max(prediction[:, 2], target[:, 2])
        enc_y2 = torch.max(prediction[:, 3], target[:, 3])
        enc_diag = ((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2).clamp(min=1e-7)

        # aspect ratio penalty, hopefully
        aspect_ratio_penalty = (4 / (math.pi ** 2)) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
        with torch.no_grad():
            alpha = aspect_ratio_penalty / (1 - iou + aspect_ratio_penalty + 1e-7)

        ciou = iou - (center_dist / enc_diag + alpha * aspect_ratio_penalty)
        loss = 1 - ciou
        return loss.mean()


class CompositeLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ciou_loss = CIoULoss()
        self.giou_loss = GIoULoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, class_predictions, objectness_predictions, localization_predictions,
                class_targets, objectness_targets, localization_targets):
        
        class_loss_weight = 0.5
        objectness_loss_weight = 0.8
        noobject_loss_weight = 0.5
        localization_loss_weight = 5

        # Class loss
        class_loss = self.bce_loss(class_predictions, class_targets)

        # Objectness loss (split into object and no-object losses)
        obj_mask = objectness_targets == 1
        noobj_mask = objectness_targets == 0

        obj_loss = self.bce_loss(objectness_predictions[obj_mask], objectness_targets[obj_mask]) if obj_mask.any() else 0
        noobj_loss = self.bce_loss(objectness_predictions[noobj_mask], objectness_targets[noobj_mask]) if noobj_mask.any() else 0

        # Convert center-based boxes to corner boxes for CIoU
        cx, cy, w, h = localization_predictions.unbind(-1)
        pred_x1 = cx - w / 2
        pred_y1 = cy - h / 2
        pred_x2 = cx + w / 2
        pred_y2 = cy + h / 2
        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

        cx, cy, w, h = localization_targets.unbind(-1)
        tgt_x1 = cx - w / 2
        tgt_y1 = cy - h / 2
        tgt_x2 = cx + w / 2
        tgt_y2 = cy + h / 2
        target_boxes = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1)

        # Only compute CIoU loss where there is an object
        if obj_mask.any():
            loc_loss = self.ciou_loss(pred_boxes[obj_mask], target_boxes[obj_mask])
        else:
            loc_loss = 0

        # Final combined loss
        weighted_loss = (
            class_loss * class_loss_weight +
            obj_loss * objectness_loss_weight +
            noobj_loss * noobject_loss_weight +
            loc_loss * localization_loss_weight
        )

        return weighted_loss