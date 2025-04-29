import torch

def generate_targets_old(raw_preds: torch.Tensor, raw_targets: list, number_of_bboxes):
    """
    Args:
        raw_preds: [B, S, S, b*5+c] – predicted tensor
        raw_targets: list of B elements, each containing a list of [class, x, y, w, h]
                    where x, y, w, h are normalized to [0,1]
    Returns:
        torch.Tensor: A tensor of shape (B, S, S, B*5 + C) representing the training targets.
    """
    B, S, _, C = raw_preds.shape
    targets = torch.zeros((B, S, S, C), device=raw_preds.device)


    for batch_idx, targets_batch in enumerate(raw_targets):
        for box in targets_batch:
            class_idx, x, y, w, h = box

            grid_x = int(x * S)
            grid_y = int(y * S)

            x_offset = x * S - grid_x
            y_offset = y * S - grid_y

            # Encode each bounding box predictor
            for b in range(number_of_bboxes):
                targets[batch_idx, grid_y, grid_x, b*5: b*5+5] = torch.tensor(
                    [x_offset, y_offset, w, h, 1], device=raw_preds.device
                )
            # Class probabilities (one-hot)
            targets[batch_idx, grid_y, grid_x, number_of_bboxes*5+int(class_idx)] = 1
    return targets

def generate_targets(raw_preds: torch.Tensor, raw_targets: list, number_of_bboxes):
    B, S, _, C = raw_preds.shape
    targets = torch.zeros((B, S, S, C), device=raw_preds.device)

    for batch_idx, targets_batch in enumerate(raw_targets):
        for box in targets_batch:
            class_idx, x, y, w, h = box

            grid_x = int(x * S)
            grid_y = int(y * S)

            x_offset = x * S - grid_x
            y_offset = y * S - grid_y

            # --- Responsibility assignment ---
            # For each bbox predictor, get the predicted box at this cell
            pred_boxes = []
            for b in range(number_of_bboxes):
                pred = raw_preds[batch_idx, grid_y, grid_x, b*5:b*5+4]
                pred_boxes.append(pred)
            gt_box = torch.tensor([x_offset, y_offset, w, h], device=raw_preds.device)

            # Compute IoU between gt_box and each pred_box
            def box_iou(box1, box2):
                # box: [x, y, w, h] (center format, normalized)
                x1_min = box1[0] - box1[2] / 2
                y1_min = box1[1] - box1[3] / 2
                x1_max = box1[0] + box1[2] / 2
                y1_max = box1[1] + box1[3] / 2

                x2_min = box2[0] - box2[2] / 2
                y2_min = box2[1] - box2[3] / 2
                x2_max = box2[0] + box2[2] / 2
                y2_max = box2[1] + box2[3] / 2

                inter_xmin = max(x1_min, x2_min)
                inter_ymin = max(y1_min, y2_min)
                inter_xmax = min(x1_max, x2_max)
                inter_ymax = min(y1_max, y2_max)

                inter_w = max(0, inter_xmax - inter_xmin)
                inter_h = max(0, inter_ymax - inter_ymin)
                inter_area = inter_w * inter_h

                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)
                union = area1 + area2 - inter_area
                if union == 0:
                    return 0.0
                return inter_area / union

            ious = [box_iou(gt_box, pred_box) for pred_box in pred_boxes]
            responsible_b = int(torch.tensor(ious).argmax())

            # Assign only the responsible predictor
            targets[batch_idx, grid_y, grid_x, responsible_b*5: responsible_b*5+5] = torch.tensor(
                [x_offset, y_offset, w, h, 1], device=raw_preds.device
            )
            # All other predictors: confidence = 0 (already zero by default)

            # Class probabilities (one-hot)
            targets[batch_idx, grid_y, grid_x, number_of_bboxes*5+int(class_idx)] = 1
    return targets

# def yolo_loss(preds: torch.Tensor, targets: torch.Tensor, number_of_bboxes: int, lambda_coord = 5, lambda_noobj = 0.5):

#     bbox_preds = preds[..., :number_of_bboxes*5]
#     bbox_targets = targets[..., :number_of_bboxes*5] 

#     cls_preds = preds[...,number_of_bboxes*5:]
#     cls_targets = targets[...,number_of_bboxes*5:]

#     obj_mask = targets[..., 4] == 1
#     noobj_mask = targets[..., 4] == 0

#     bbox_loss = torch.sum((bbox_targets[obj_mask]-bbox_preds[obj_mask])**2)

#     cls_obj_loss = torch.sum((cls_targets[obj_mask]-cls_preds[obj_mask])**2)

    



#     print("allo")


def yolo_loss(preds: torch.Tensor, targets: torch.Tensor, number_of_bboxes: int, 
              lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLOv1 loss as per the original paper's equation.
    
    Args:
        preds: Tensor of shape (batch, S, S, B*5 + C)
        targets: Tensor of same shape as preds
        number_of_bboxes: B (number of bounding box predictors per cell)
        lambda_coord: Weight for coordinate loss (default: 5)
        lambda_noobj: Weight for no-object confidence loss (default: 0.5)
    
    Returns:
        Total YOLOv1 loss
    """
    # Split predictions and targets
    bbox_preds = preds[..., :number_of_bboxes*5].reshape(*preds.shape[:3], number_of_bboxes, 5)
    bbox_targets = targets[..., :number_of_bboxes*5].reshape(*targets.shape[:3], number_of_bboxes, 5)
    
    cls_preds = preds[..., number_of_bboxes*5:]
    cls_targets = targets[..., number_of_bboxes*5:]
    
    # Object masks (I^obj in the paper)
    obj_mask = bbox_targets[..., 4] == 1  # Shape: (batch, S, S, B)
    noobj_mask = bbox_targets[..., 4] == 0
    
    # --- 1. Coordinate loss (x,y) ---
    xy_pred = bbox_preds[..., :2]  # (batch, S, S, B, 2)
    xy_target = bbox_targets[..., :2]
    xy_loss = lambda_coord * torch.sum(
        obj_mask.unsqueeze(-1) * (xy_pred - xy_target).pow(2))
    
    # --- 2. Coordinate loss (sqrt(w), sqrt(h)) ---
    wh_pred = bbox_preds[..., 2:4].clamp(min=1e-6).sqrt()  # prevent sqrt(negative)
    wh_target = bbox_targets[..., 2:4].sqrt()
    wh_loss = lambda_coord * torch.sum(
        obj_mask.unsqueeze(-1) * (wh_pred - wh_target).pow(2))
    
    # --- 3. Object confidence loss ---
    conf_pred = bbox_preds[..., 4]  # (batch, S, S, B)
    conf_target = bbox_targets[..., 4]
    obj_conf_loss = torch.sum(obj_mask * (conf_pred - conf_target).pow(2))
    
    # --- 4. No-object confidence loss ---
    noobj_conf_loss = lambda_noobj * torch.sum(
        noobj_mask * (conf_pred - conf_target).pow(2))
    
    # --- 5. Class probability loss ---
    # Only compute for cells that contain objects (I^obj_i in paper)
    cls_loss = torch.sum(
        obj_mask.any(-1, keepdim=True) * (cls_preds - cls_targets).pow(2))
    
    # Sum all components
    total_loss = (xy_loss + wh_loss + obj_conf_loss + noobj_conf_loss + cls_loss)
    
    return total_loss / preds.shape[0]  # Average over batch