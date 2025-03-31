import torch
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np

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

def calculate_yolo_loss_v3(predictions: torch.Tensor, targets: torch.Tensor, 
                        localization_loss_weight=5, no_obj_loss_weight=0.5):
    """
    Optimized YOLO loss function that properly handles nested tensor dimensions.
    
    Args:
        predictions: Shape [batch_size, grid_size, grid_size, num_anchors, 5+num_classes]
        targets: Shape [batch_size, grid_size, grid_size, num_anchors, 5+num_classes]
    """
    batch_size = predictions.shape[0]
    device = predictions.device
    total_loss = torch.tensor(0.0, device=device)
    
    # Process batch items individually
    for b in range(batch_size):
        pred = predictions[b]  # [grid_size, grid_size, num_anchors, 5+num_classes]
        targ = targets[b]      # [grid_size, grid_size, num_anchors, 5+num_classes]
        
        # Extract box predictions and targets
        pred_boxes = pred[..., :4]        # [grid_size, grid_size, num_anchors, 4]
        pred_conf = pred[..., 4]          # [grid_size, grid_size, num_anchors]
        pred_classes = pred[..., 5:]      # [grid_size, grid_size, num_anchors, num_classes]
        
        targ_boxes = targ[..., :4]        # [grid_size, grid_size, num_anchors, 4]
        targ_conf = targ[..., 4]          # [grid_size, grid_size, num_anchors]
        targ_classes = targ[..., 5:]      # [grid_size, grid_size, num_anchors, num_classes]
        
        # Create masks for cells with and without objects
        obj_mask = (targ_conf > 0.5)      # [grid_size, grid_size, num_anchors]
        no_obj_mask = (targ_conf < 0.5)   # [grid_size, grid_size, num_anchors]
        
        # 1. Box coordinate loss (only for cells with objects)
        box_loss = 0
        if obj_mask.sum() > 0:
            # MSE loss on box coordinates where objects exist
            box_loss = localization_loss_weight * torch.sum(
                (pred_boxes[obj_mask] - targ_boxes[obj_mask])**2
            )
        
        # 2. Object confidence loss (cells with objects)
        obj_loss = 0
        if obj_mask.sum() > 0:
            obj_loss = torch.sum((pred_conf[obj_mask] - targ_conf[obj_mask])**2)
        
        # 3. No-object confidence loss (cells without objects)
        no_obj_loss = 0
        if no_obj_mask.sum() > 0:
            no_obj_loss = no_obj_loss_weight * torch.sum(pred_conf[no_obj_mask]**2)
        
        # 4. Class prediction loss (only for cells with objects)
        class_loss = 0
        if obj_mask.sum() > 0:
            # Get class predictions and targets for cells with objects
            obj_pred_classes = pred_classes[obj_mask]  # [num_objects, num_classes]
            obj_targ_classes = targ_classes[obj_mask]  # [num_objects, num_classes]
            
            # Get class indices from one-hot encoded targets
            targ_class_idx = torch.argmax(obj_targ_classes, dim=1)
            
            # Use cross-entropy loss on class predictions
            class_loss = f.cross_entropy(obj_pred_classes, targ_class_idx)
        
        # Sum all loss components
        batch_loss = box_loss + obj_loss + no_obj_loss + class_loss
        total_loss += batch_loss
    
    # Average loss over batch
    return total_loss / batch_size

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
              
        # Compute the width and height adjustments (scaled relative to anchor size)
        target_width = width * grid_size / anchor_width
        target_height = height * grid_size / anchor_height

        # Assign the bounding box offsets and size adjustments
        target[grid_y, grid_x,  best_anchor_index, :4] = torch.tensor([x_offset, y_offset, target_width, target_height])
        
        # Set objectness score to 1 (since this anchor box has a corresponding ground truth box)
        target[grid_y, grid_x, best_anchor_index, 4] = 1
        
        # One-hot encode the class label
        target[grid_y, grid_x, best_anchor_index, 5 + int(class_label)] = 1
    return target

class ConvBlock(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)

class YOLOHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Add adaptive pooling to get to grid_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.detector = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.detector(x)
        # Reshape to include the anchor boxes dimension
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, height, width, self.num_anchors, 5 + self.num_classes)
        return x
    
class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
        super(YOLO, self).__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, grid_size, num_classes, anchors, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.image_dir)]
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.anchors = anchors

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = read_image(image_path)
        
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"  
        label_path = os.path.join(self.label_dir, label_file)
        
        if os.path.exists(label_path):
            bounding_boxes = load_yolo_label(label_path)
            target = create_yolo_target(bounding_boxes, self.grid_size, self.num_classes, self.anchors)
        else:
            #Label could not be found -> Using default zero tensor
            target = torch.zeros(self.grid_size, self.grid_size, len(self.anchors), 5 + self.num_classes)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target