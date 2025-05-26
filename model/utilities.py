import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
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

    def forward(self, class_predictions, objectness_predictions, localization_predictions, targets, device):
        class_loss_weight = 1.2
        objectness_loss_weight = 0.9
        noobject_loss_weight = 0.5
        localization_loss_weight = 4

        batch_size, grid_size, _, num_classes = class_predictions.shape

        class_loss = 0
        obj_loss = 0
        noobj_loss = 0
        localization_loss = 0

        full_objectness_target_batch = torch.zeros_like(objectness_predictions)

        for index in range(batch_size):
            target = targets[index]
            if len(target) == 0:
                continue

            class_targets = target[:, 0].long()
            full_objectness_target = torch.zeros((grid_size, grid_size), device=device)
            full_class_target = torch.zeros((grid_size, grid_size, num_classes), device=device)
            localization_targets = target[:, 1:]

            grid_x = (target[:, 1] * grid_size).long()
            grid_y = (target[:, 2] * grid_size).long()

            full_objectness_target[grid_y, grid_x] = 1
            full_class_target[grid_y, grid_x, class_targets] = 1
            full_objectness_target_batch[index] = full_objectness_target

            selected_prediction = class_predictions[index, grid_y, grid_x]
            selected_target = full_class_target[grid_y, grid_x]

            localization_box = localization_predictions[index, grid_y, grid_x]
            cx, cy, w, h = localization_box[..., 0], localization_box[..., 1], localization_box[..., 2], localization_box[..., 3]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            localization_box = torch.stack([x1, y1, x2, y2], dim=-1)

            cx, cy, w, h = localization_targets[..., 0], localization_targets[..., 1], localization_targets[..., 2], localization_targets[..., 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            localization_targets = torch.stack([x1, y1, x2, y2], dim=-1)

            # aspect_ratio = torch.max( w/ h, h / w)
            # aspect_ratio_penalty = torch.mean(aspect_ratio)
            # localization_loss += 0.01 * aspect_ratio_penalty

            localization_loss += self.ciou_loss(localization_box, localization_targets)
            class_loss += self.bce_loss(selected_prediction, selected_target)

        obj_mask = full_objectness_target_batch == 1
        noobj_mask = full_objectness_target_batch == 0

        if obj_mask.any():
            obj_loss = self.bce_loss(objectness_predictions[obj_mask], full_objectness_target_batch[obj_mask])
        if noobj_mask.any():
            noobj_loss = self.bce_loss(objectness_predictions[noobj_mask], full_objectness_target_batch[noobj_mask])

        num_targets = sum(len(target) for target in targets if len(target) > 0)
        if num_targets > 0:
            class_loss /= num_targets
            localization_loss /= num_targets


        weighted_loss = (
            (class_loss * class_loss_weight)
            + (obj_loss * objectness_loss_weight)
            + (noobj_loss * noobject_loss_weight)
            + (localization_loss * localization_loss_weight)
        )

        return weighted_loss
