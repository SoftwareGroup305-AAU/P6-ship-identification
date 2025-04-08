import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, images):
        return self.relu(self.bn(self.conv(images)))

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, images):
        return self.layers(images)

class Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(256, 128, 3, padding=1)
        self.classification = nn.Conv2d(128, num_classes, 1)
        self.objectness = nn.Conv2d(128, 1, 1)
        self.localization = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        classification_output = (self.classification(x)).permute(0, 2, 3, 1).contiguous() # removed activation function, dont know if good or not
        objectness_output = F.sigmoid(self.objectness(x)).squeeze(1)
        localization_output = F.sigmoid(self.localization(x)).permute(0, 2, 3, 1).contiguous()
        return classification_output, objectness_output, localization_output

class YOLO(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head(num_classes)
    
    def forward(self, images):
        feature_map = self.backbone(images)
        return self.head(feature_map)