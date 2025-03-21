from typing import Callable
from PIL import Image
import torch
from torchvision import transforms
import urllib
import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models.detection as detection
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

def calculate_yolo_loss(predictions: torch.Tensor, targets: torch.Tensor, localization_loss_weight = 5, no_obj_confidence_loss_weight = 0.5):

    prediction_boxes = predictions[..., :4]
    prediction_confidence = predictions[...,4]
    prediction_classes = predictions[...,5:]
    
    target_boxes = targets[..., :4]
    target_confidence = targets[...,4]
    target_classes = targets[...,5:]
    
    #iou scores between predicted boxes and target boxes (used for loss calc)
    # Does NOT work because it does not get a single box, but multiple. 
    iou_scores = torch.stack([calculate_iou(prediction_boxes[i], target_boxes[i]) for i in range(predictions.shape[0])])
    
    #calculate losses
    box_loss = localization_loss_weight * torch.sum((prediction_boxes - target_boxes) ** 2)
    obj_loss = f.binary_cross_entropy(prediction_confidence, target_confidence)
    no_obj_loss = no_obj_confidence_loss_weight * torch.sum((prediction_confidence[target_confidence == 0]) ** 2)
    class_loss = f.cross_entropy(prediction_classes, target_classes)

    total_loss = box_loss + obj_loss + no_obj_loss + class_loss
    return total_loss

def calculate_yolo_loss_v2(predictions, targets, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLO loss.
    - predictions: Predicted tensor.
    - targets: Ground truth tensor.
    """
    # Unpack predictions and targets
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
    noobj_loss = lambda_noobj * torch.sum((pred_conf[target_conf == 0]) ** 2)

    # Classification Loss
    class_loss = torch.sum((pred_classes - target_classes) ** 2)

    # Total Loss
    total_loss = box_loss + obj_loss + noobj_loss + class_loss
    return total_loss

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

# def import_data():
#     transform = transforms.Compose([
#         transforms.Resize(300, 300),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), 
#                             (0.5, 0.5, 0.5))
#     ])

#     training_set = 0 # training data location
#     training_data_loader = dataloader(training_set, batch_size=32, shuffle=True)
    
#     test_set = 0 # test data location
#     test_data_loader = dataloader(test_set, batch_size=32, shuffle=False)


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


# class YOLO(nn.Module):
#     def __init__(self, num_classes, num_anchors, grid_size):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_anchors = num_anchors
#         self.grid_size = grid_size

#         #Backbone
#         self.backbone = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         #Detection head
#         self.detector = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(256 * (grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes))
#         )
    
#     def forward(self, x):
#         features = self.backbone(x)
#         predictions_flat = self.detector(features)
#         batch_size = x.shape[0]
#         predictions = predictions_flat.reshape(
#             batch_size,
#             self.grid_size,
#             self.grid_size,
#             self.num_anchors,
#             5+self.num_classes
#         )
#         return predictions

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

# class Training():
#     num_epochs = 10 #Number of passes over training data
#     model.train()
#     optimizer = torch.optim.SGD(model.parameters, lr=0.005)
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         loss_dict = model(images, targets)

#         losses = sum(loss for loss in loss_dict.values())
#         losses.backward()
#         optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {losses.item():.4f}")
# # from PIL import Image
# # import torch
# # from torchvision import transforms
# # import urllib
# # import torch

# torch.nn.Conv2d(stride=10, padding='valid', dilation=5, groups=4)We should probably look at using these params, "groups greater than 1, allows for specialization and just a tad performance"
