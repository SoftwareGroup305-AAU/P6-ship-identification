import torch
import torch.nn as nn
import torch.nn.functional as F
    
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
        giou = giou.clamp(min=0, max=1) # limits negative giou values, should probably handle this better
        return 1 - giou.mean()
    
class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.giou_loss = GIoULoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, class_predictions, objectness_predictions, localization_predictions, targets):
        # no clue if weights should be adjusted and what benefit this would provide
        class_loss_weight = 1
        objectness_loss_weight = 1
        localization_loss_weight = 1

        batch_size, grid_size, grid_size, num_classes = class_predictions.shape

        class_loss = 0
        objectness_loss = 0
        localization_loss = 0


        for index in range(batch_size):
            target = targets[index]

            if len(target) == 0:
                continue

            class_targets = target[:, 0].long()
            objectness_targets = torch.ones(len(target), device=objectness_predictions.device)
            localization_targets = target[:, 1:]

            grid_x = (target[:, 1] * grid_size).long()
            grid_y = (target[:, 2] * grid_size).long()

            slelected_prediction = class_predictions[index, grid_y, grid_x]

            localization_box = localization_predictions[index, grid_y, grid_x]

            center_x, center_y, width, height = localization_box[..., 0], localization_box[..., 1], localization_box[..., 2], localization_box[..., 3]
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            localization_box = torch.stack([x1, y1, x2, y2], dim=-1)

            center_x, center_y, width, height = localization_targets[..., 0], localization_targets[..., 1], localization_targets[..., 2], localization_targets[..., 3]
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            localization_targets = torch.stack([x1, y1, x2, y2], dim=-1)

            class_loss += self.ce_loss(slelected_prediction, class_targets)
            objectness_loss += self.mse_loss(objectness_predictions[index, grid_y, grid_x], objectness_targets.squeeze(-1))
            localization_loss = self.giou_loss(localization_box, localization_targets)

        num_targets = sum(len(target) for target in targets if len(target) > 0)
        if num_targets > 0:
            class_loss /= num_targets
            objectness_loss /= num_targets
            localization_loss /= num_targets

        weighted_loss = (class_loss * class_loss_weight) + (objectness_loss * objectness_loss_weight) + (localization_loss * localization_loss_weight)
        return weighted_loss