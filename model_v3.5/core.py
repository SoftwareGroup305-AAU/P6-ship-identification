import torch
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
        self.layer1 = ConvBlock(3, 16, 3, 1, 1)
        self.layer2 = ConvBlock(16, 32, 3, 2, 1)  # Downsample
        self.layer3 = ConvBlock(32, 64, 3, 2, 1)  # Downsample
        self.layer4 = ConvBlock(64, 128, 3, 2, 1) # Downsample
        self.layer5 = ConvBlock(128, 256, 3, 2, 1) # Downsample

    def forward(self, images):
        x = self.layer1(images)
        p3 = self.layer2(x)
        p5 = self.layer3(p3)
        p7 = self.layer4(p5)
        p7 = self.layer5(p7)
        return p3, p5, p7

class Head(nn.Module):
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.shared_conv = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.cls_conv = nn.Conv2d(128, num_classes, 1)
        self.obj_conv = nn.Conv2d(128, 1, 1)
        self.reg_conv = nn.Conv2d(128, 4 * reg_max, 1)

    def forward(self, x):
        x = F.relu(self.shared_conv(x))
        cls_logits = self.cls_conv(x).permute(0, 2, 3, 1).contiguous()
        obj_logits = self.obj_conv(x).sigmoid().permute(0, 2, 3, 1).contiguous()
        bbox_logits = self.reg_conv(x)
        return {'cls': cls_logits, 'obj': obj_logits, 'bbox': bbox_logits}

class YOLO(nn.Module):
    def __init__(self, num_classes, reg_max=16):
        super().__init__()
        self.backbone = Backbone()
        self.head_p3 = Head(32, num_classes, reg_max)
        self.head_p5 = Head(64, num_classes, reg_max)
        self.head_p7 = Head(256, num_classes, reg_max)

    def forward(self, images):
        p3, p5, p7 = self.backbone(images)
        return {
            'p3': self.head_p3(p3),
            'p5': self.head_p5(p5),
            'p7': self.head_p7(p7),
        }
