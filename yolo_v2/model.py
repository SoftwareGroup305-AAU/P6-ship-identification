import torch.nn as nn

class YOLOModel(nn.Module):
    def __init__(self, grid_size=7, num_classes=2):
        super(YOLOModel, self).__init__()
        self.backbone = Backbone()
        self.head = Head(grid_size, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)

class Head(nn.Module):
    def __init__(self, grid_size, num_classes):
        super(Head, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
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
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))