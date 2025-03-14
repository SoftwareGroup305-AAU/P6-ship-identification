from PIL import Image
import torch
from torchvision import transforms
import urllib
import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import torchvision.models.detection as detection

def calculate_iou(box_1, box_2) -> float:
    """
    Calculates the Intersection over Union (IoU) between two boxes.

    Parameters:
        box_1 (tuple): (center_x, center_y, width, height) of the first box.
        box_2 (tuple): (center_x, center_y, width, height) of the second box.

    Returns:
        float: IoU value between 0 and 1.
    """
    center_x1, center_y1, width_1, height_1 = box_1
    center_x2, center_y2, width_2, height_2 = box_2

    box_1_area = width_1*height_1
    box_2_area = width_2*height_2

    box_1_left = center_x1 - (width_1 / 2)
    box_1_right = center_x1 + (width_1 / 2)
    box_1_bottom = center_y1 - (height_1 / 2)
    box_1_top = center_y1 + (height_1 / 2)

    box_2_left = center_x2 - (width_2 / 2)
    box_2_right = center_x2 + (width_2 / 2)
    box_2_bottom = center_y2 - (height_2 / 2)
    box_2_top = center_y2 + (height_2 / 2)

    intersection_width = min(box_1_right, box_2_right) - max(box_1_left, box_2_left)
    intersection_height = min(box_1_top, box_2_top) -max(box_1_bottom, box_2_bottom)

    if intersection_width <= 0 or intersection_height <= 0:
        return 0

    intersection_area = intersection_width*intersection_height
    union_area = box_1_area+box_2_area-intersection_area

    iou = intersection_area / union_area
    return iou

def create_target(bounding_boxes, grid_size, num_classes, anchors):
    """
    creates target tensor

    Parameters:
        bounding_boxes: list of bounding boxes (center_x, center_y, width, height)
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

            iou = calculate_iou((0, 0, anchor_width, anchor_height), (0, 0, width, height))

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
