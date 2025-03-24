import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x1, box1_y1 = x1 - w1 / 2, y1 - h1 / 2
    box1_x2, box1_y2 = x1 + w1 / 2, y1 + h1 / 2
    box2_x1, box2_y1 = x2 - w2 / 2, y2 - h2 / 2
    box2_x2, box2_y2 = x2 + w2 / 2, y2 + h2 / 2

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)

def create_yolo_target(bounding_boxes, grid_size, num_classes, anchors):
    """
    creates target tensor

    Parameters:
        bounding_boxes: list of ground truth bounding boxes (center_x, center_y, width, height)
        grid_size: size of grid
        num_classes: number of classes
        anchors: list of anchors (height, width)

    Returns:
        Tensor: target tensor.
    """
    num_anchors = len(anchors)
    target = torch.zeros(grid_size, grid_size, num_anchors, 5 + num_classes)
    
    for bounding_box in bounding_boxes:
        class_label, center_x, center_y, width, height = bounding_box 
        
        # Compute which grid cell the bounding box would be inside of
        grid_x = int(center_x * grid_size)
        grid_y = int(center_y * grid_size)
        
        # Compute the exact position within the grid cell (the offsets)
        x_offset = (center_x * grid_size) - grid_x
        y_offset = (center_y * grid_size) - grid_y
        
        # find anchor with most overlap
        max_anchor_overlap = 0
        best_anchor_index = 0
        for anchor_index in range(num_anchors):
            anchor_width, anchor_height = anchors[anchor_index]
            anchor_tensor = torch.tensor([0,0, anchor_width, anchor_height], dtype=torch.float32)
            bounding_box_tensor = torch.tensor([0, 0, width, height], dtype=torch.float32)
            iou = calculate_iou(anchor_tensor, bounding_box_tensor)
            if (iou > max_anchor_overlap):
                max_anchor_overlap = iou
                best_anchor_index = anchor_index

        anchor_width, anchor_height = anchors[best_anchor_index]
              
        # Compute the width and height offsets (logorithmic scaling)
        width_offset = math.log(width / anchor_width)
        height_offset = math.log(height / anchor_height)  

        # Assign the bounding box offsets and size adjustments
        target[grid_x, grid_y,  best_anchor_index, :4] = torch.tensor([x_offset, y_offset, width_offset, height_offset])
        
        # Set objectness score to 1 (since this anchor box has a corresponding ground truth box)
        target[grid_x, grid_y, best_anchor_index, 4] = 1
        
        # One-hot encode the class label
        target[grid_y, grid_x, best_anchor_index, 5 + int(class_label)] = 1
    return target

def generate_anchors(scales, ratios):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return np.array(anchors)

def load_yolo_label(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data.columns = ["class", "center_x", "center_y", "width", "height"]
    bounding_boxes = []
    for index, row in data.iterrows():
        bounding_boxes.append([int(row["class"]), 
                               float(row["center_x"]), 
                               float(row["center_y"]), 
                               float(row["width"]), 
                               float(row["height"])])
    return bounding_boxes


def yolo_loss(predictions, targets, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLO loss.
    - predictions: Predicted tensor.
    - targets: Ground truth tensor.
    """
    # Unpack predictions and targets
    
    assert predictions.shape == targets.shape

    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]
    target_boxes = targets[..., :4]
    target_conf = targets[..., 4]
    target_classes = targets[..., 5:]
    
    # Localization Loss
    box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

    # Confidence Loss
    obj_loss = torch.sum((pred_conf - target_conf) ** 2)

    # Classification Loss
    class_loss = torch.sum((pred_classes - target_classes) ** 2)

    # Total Loss
    total_loss = box_loss + obj_loss + class_loss
    return total_loss