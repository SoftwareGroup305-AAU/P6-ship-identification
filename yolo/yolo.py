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


def calculate_yolo_loss(predictions: torch.Tensor, targets: torch.Tensor, localization_loss_weight = 5, no_obj_confidence_loss_weight = 0.5):

    prediction_boxes = predictions[..., :4]
    prediction_confidence = predictions[...,4]
    prediction_classes = predictions[...,5:]
    
    target_boxes = targets[..., :4]
    target_confidence = targets[...,4]
    target_classes = targets[...,5:]
    
    #iou_scores between predicted boxes and target boxes (used for loss calc)
    iou_scores = torch.stack([calculate_iou(prediction_boxes[i], target_boxes[i]) for i in range(predictions.shape[0])])

    #calculate losses
    box_loss = localization_loss_weight * torch.mean(iou_scores)
    obj_loss = f.binary_cross_entropy(prediction_confidence, target_confidence)
    no_obj_loss = no_obj_confidence_loss_weight * torch.sum((prediction_confidence[target_confidence == 0]) ** 2)
    class_loss = f.cross_entropy(prediction_classes, target_classes)

    total_loss = box_loss + obj_loss + no_obj_loss + class_loss
    return total_loss


def create_target(bounding_boxes, grid_size, num_classes, anchors):
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
        target_width = torch.log(width * grid_size / anchor_width)
        target_height = torch.log(height * grid_size / anchor_height)

        # Assign the bounding box offsets and size adjustments
        target[grid_y, grid_x,  best_anchor_index, :4] = torch.tensor([x_offset, y_offset, target_width, target_height])
        
        # Set objectness score to 1 (since this anchor box has a corresponding ground truth box)
        target[grid_y, grid_x, best_anchor_index, 4] = 1
        
        # One-hot encode the class label
        target[grid_y, grid_x, best_anchor_index, 5 + int(class_label)] = 1
    return target

def import_data():
    transform = transforms.Compose([
        transforms.Resize(300, 300),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                            (0.5, 0.5, 0.5))
    ])

    training_set = 0 # training data location
    training_data_loader = dataloader(training_set, batch_size=32, shuffle=True)
    
    test_set = 0 # test data location
    test_data_loader = dataloader(test_set, batch_size=32, shuffle=False)

class YOLO(nn.Module):
    def __init__(self, num_classes, num_anchors, grid_size):
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size

        #Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(64),
        )
        #Detection head
        self.detector = nn.Sequential(
            nn.Conv2d(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*(grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes))
        )
    def forward(self, image):
        features = self.backbone(image)
        predictions = self.detector(features)
        # handle prediction


class Training():
    num_epochs = 10 #Number of passes over training data
    model.train()
    optimizer = torch.optim.SGD(model.parameters, lr=0.005)
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {losses.item():.4f}")
# from PIL import Image
# import torch
# from torchvision import transforms
# import urllib
# import torch

# torch.nn.Conv2d(stride=10, padding='valid', dilation=5, groups=4)We should probably look at using these params, "groups greater than 1, allows for specialization and just a tad performance"
