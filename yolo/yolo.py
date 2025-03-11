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

model = detection.ssd300_vgg16(pretrained=True)#We use pretrained, and then specialize it for our purpose with more training

def create_target(bounding_boxes, grid_size, num_classes, anchors):
    num_anchors = len(anchors)
    """
    Creates a zeroed 4D tensor representing the target values for each grid cell:
    1st dimension: grid_size
    2nd dimension: grid_size
    3rd dimension: num_anchors
    4th dimension: Bounding box attributes (tx, ty, tw, th, objectness) + class probabilities for each class (5 + num_classes)
    """
    target = torch.zeros(grid_size, grid_size, num_anchors, 5 + num_classes)
    
    for bounding_box in bounding_boxes:
        class_label, center_x, center_y, width, height = bounding_box 
        
        # Compute which grid cell the bounding box would be inside of
        grid_x = int(center_x * grid_size)
        grid_y = int(center_y * grid_size)
        
        # Compute the exact position within the grid cell (the offsets)
        x_offset = (center_x * grid_size) - grid_x
        y_offset = (center_y * grid_size) - grid_y
        
        for anchor_index in range(num_anchors):  # Iterate over all anchor boxes
            anchor_width, anchor_height = anchors[anchor_index]
            
            # Compute the width and height adjustments (scaled relative to anchor size)
            target_width = torch.log(width * grid_size / anchor_width)
            target_height = torch.log(height * grid_size / anchor_height)
            
            # Assign the bounding box offsets and size adjustments
            target[grid_y, grid_x,  anchor_index, :4] = torch.tensor([x_offset, y_offset, target_width, target_height])
            
            # Set objectness score to 1 (since this anchor box has a corresponding ground truth box)
            target[grid_y, grid_x, anchor_index, 4] = 1
            
            # One-hot encode the class label
            target[grid_y, grid_x, anchor_index, 5 + int(class_label)] = 1
    return target


def convert_labels_to_target(bounding_boxes, grid_size, num_classes, num_anchors):
    target = torch.zeros((grid_size, grid_size, num_anchors, 5 + num_classes))  # (tx, ty, tw, th, obj, class_probs)

    for bbox in bounding_boxes:
        cls, cx, cy, w, h = bbox

        grid_x = int(cx * grid_size)
        grid_y = int(cy * grid_size)

        tx = cx * grid_size - grid_x
        ty = cy * grid_size - grid_y

        target[grid_y, grid_x, 0, :4] = torch.tensor([tx, ty, w, h])  # Assume 1 anchor for simplicity
        target[grid_y, grid_x, 0, 4] = 1  # Objectness score
        target[grid_y, grid_x, 0, 5 + int(cls)] = 1  # One-hot class

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