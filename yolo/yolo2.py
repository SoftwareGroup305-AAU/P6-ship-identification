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
from utils import create_yolo_target, load_yolo_label
import cv2


class ConvBlock(nn.Module):
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
        
        self.adpool = nn.AdaptiveMaxPool2d((grid_size, grid_size))
        
        self.detector = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.adpool(x)
        
        predictions = self.detector(x)
        
        batch_size = predictions.shape[0]
        predictions = predictions.permute(0, 2, 3, 1).contiguous()
        return predictions.view(batch_size, self.grid_size, self.grid_size, 
                              self.num_anchors, 5 + self.num_classes)

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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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
