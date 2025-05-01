import torch
import torch.nn as nn

class YOLOv1Reduced(nn.Module):
    def __init__(self, number_of_bboxes=2, number_of_classes=20):
        super(YOLOv1Reduced, self).__init__()
        self.S = 7
        self.B = number_of_bboxes
        self.C = number_of_classes

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.conv_layers = nn.Sequential(
            conv_block(3, 32, 7, 2, 3), 
            nn.MaxPool2d(2, 2),

            conv_block(32, 96, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(96, 64, 1, 1, 0),
            conv_block(64, 128, 3, 1, 1),
            conv_block(128, 128, 1, 1, 0),
            conv_block(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(512, 512, 1, 1, 0),
            conv_block(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 1, 1)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)

