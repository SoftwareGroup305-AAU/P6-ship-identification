import torch
import torch.nn.functional as F
import torch.nn as nn

def bbox_attr(data, i, num_classes):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = num_classes + i
    return data[..., attr_start::5]

def bbox_to_coords(t, num_classes):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2, num_classes)
    x = bbox_attr(t, 0, num_classes)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3, num_classes)
    y = bbox_attr(t, 1, num_classes)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)



def get_iou(p, a, num_predictors, num_classes):
    p_tl, p_br = bbox_to_coords(p, num_classes)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a, num_classes)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, num_predictors, num_predictors, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2, num_classes) * bbox_attr(p, 3, num_classes)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2, num_classes) * bbox_attr(a, 3, num_classes)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = 1E-6
    intersection[zero_unions] = 0.0

    return intersection / union


def sum_square_error_loss(preds: torch.Tensor, targets: torch.Tensor, 
                         num_predictors: int, num_classes: int,
                         lambda_coord=5, lambda_noobj=0.5):
    """
    Improved YOLOv1 loss function matching the better class implementation
    
    Args:
        preds: Tensor of shape (batch, S, S, B*5 + C)
        targets: Tensor of same shape as preds
        number_of_bboxes: B (number of bounding box predictors per cell)
        num_classes: C (number of classes)
        lambda_coord: Weight for coordinate loss (default: 5)
        lambda_noobj: Weight for no-object confidence loss (default: 0.5)
    
    Returns:
        Total YOLOv1 loss
    """
    batch_size = preds.shape[0]
    
    # Reshape predictions and targets
    bbox_preds = preds[..., :num_predictors*5].reshape(*preds.shape[:3], num_predictors, 5)
    bbox_targets = targets[..., :num_predictors*5].reshape(*targets.shape[:3], num_predictors, 5)
    
    cls_preds = preds[..., num_predictors*5:]
    cls_targets = targets[..., num_predictors*5:]
    
    # Calculate IoU between predictions and targets
    iou = get_iou(preds, targets, num_predictors, num_classes)  # Shape: (batch, S, S, B, B)
    max_iou = torch.max(iou, dim=-1)[0]  # (batch, S, S, B)
    
    # Create masks
    obj_mask = bbox_targets[..., 4] > 0.0  # Cells with objects
    responsible_mask = torch.zeros_like(obj_mask).scatter_(
        -1, 
        torch.argmax(max_iou, dim=-1, keepdim=True),
        value=1
    )
    obj_responsible_mask = obj_mask * responsible_mask
    noobj_mask = ~obj_responsible_mask
    
    # 1. XY position loss
    xy_pred = bbox_preds[..., :2]
    xy_target = bbox_targets[..., :2]
    xy_loss = F.mse_loss(
        obj_responsible_mask.unsqueeze(-1) * xy_pred,
        obj_responsible_mask.unsqueeze(-1) * xy_target,
        reduction='sum'
    )
    
    # 2. WH dimension loss (with sqrt)
    wh_pred = torch.sign(bbox_preds[..., 2:4]) * torch.sqrt(torch.abs(bbox_preds[..., 2:4]) + 1e-6)
    wh_target = torch.sqrt(bbox_targets[..., 2:4])
    wh_loss = F.mse_loss(
        obj_responsible_mask.unsqueeze(-1) * wh_pred,
        obj_responsible_mask.unsqueeze(-1) * wh_target,
        reduction='sum'
    )
    
    # 3. Object confidence loss (target is 1 for responsible predictors)
    obj_conf_loss = F.mse_loss(
        obj_responsible_mask * bbox_preds[..., 4],
        obj_responsible_mask * torch.ones_like(max_iou),
        reduction='sum'
    )
    
    # 4. No-object confidence loss (target is 0)
    noobj_conf_loss = F.mse_loss(
        noobj_mask * bbox_preds[..., 4],
        torch.zeros_like(max_iou),
        reduction='sum'
    )
    
    # 5. Class probability loss
    cls_loss = F.mse_loss(
        obj_mask.any(-1, keepdim=True) * cls_preds,
        obj_mask.any(-1, keepdim=True) * cls_targets,
        reduction='sum'
    )
    
    # Combine all losses with weights
    total_loss = (
        lambda_coord * (xy_loss + wh_loss) +
        obj_conf_loss +
        lambda_noobj * noobj_conf_loss +
        cls_loss
    )
    
    return total_loss / batch_size

class SumSquaredErrorLoss(nn.Module):
    def __init__(self, num_classes, num_predictors):
        super().__init__()
        self.l_coord = 5
        self.l_noobj = 0.5
        self.C = num_classes
        self.B = num_predictors

    def forward(self, p, a):

        batch_size =p.shape[0]
        # Calculate IOU of each predicted bbox against the ground truth bbox
        iou = get_iou(p, a, self.B, self.C)                     # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim=-1)[0]     # (batch, S, S, B)

        # Get masks
        bbox_mask = bbox_attr(a, 4, self.C) > 0.0
        p_template = bbox_attr(p, 4, self.C) > 0.0
        obj_i = bbox_mask[..., 0:1]         # 1 if grid I has any object at all
        responsible = torch.zeros_like(p_template).scatter_(       # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),                # (batch, S, S, B)
            value=1                         # 1 if bounding box is "responsible" for predicting the object
        )
        obj_ij = obj_i * responsible        # 1 if object exists AND bbox is responsible
        noobj_ij = ~obj_ij                  # Otherwise, confidence should be 0

        # XY position losses
        x_losses = mse_loss(
            obj_ij * bbox_attr(p, 0, self.C),
            obj_ij * bbox_attr(a, 0, self.C)
        )
        y_losses = mse_loss(
            obj_ij * bbox_attr(p, 1, self.C),
            obj_ij * bbox_attr(a, 1, self.C)
        )
        pos_losses = x_losses + y_losses
        # print('pos_losses', pos_losses.item())

        # Bbox dimension losses
        p_width = bbox_attr(p, 2, self.C)
        a_width = bbox_attr(a, 2, self.C)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + 1e-6),
            obj_ij * torch.sqrt(a_width)
        )
        p_height = bbox_attr(p, 3, self.C)
        a_height = bbox_attr(a, 3, self.C)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + 1e-6),
            obj_ij * torch.sqrt(a_height)
        )
        dim_losses = width_losses + height_losses
        # print('dim_losses', dim_losses.item())

        # Confidence losses (target confidence is IOU)
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, 4, self.C),
            obj_ij * torch.ones_like(max_iou)
        )
        # print('obj_confidence_losses', obj_confidence_losses.item())
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, 4, self.C),
            torch.zeros_like(max_iou)
        )
        # print('noobj_confidence_losses', noobj_confidence_losses.item())

        # Classification losses
        class_losses = mse_loss(
            obj_i * p[..., :self.C],
            obj_i * a[..., :self.C]
        )
        # print('class_losses', class_losses.item())

        total = self.l_coord * (pos_losses + dim_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses
        return total / batch_size


def mse_loss(a, b):
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )
