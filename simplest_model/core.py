import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, number_of_bboxes=2, number_of_classes=20):
        super(YOLOv1, self).__init__()
        self.S = 7
        self.B = number_of_bboxes
        self.C = number_of_classes

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.conv_layers = nn.Sequential(
            conv_block(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),

            conv_block(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            conv_block(192, 128, 1, 1, 0),
            conv_block(128, 256, 3, 1, 1),
            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            *[layer for _ in range(4) for layer in [
                conv_block(512, 256, 1, 1, 0),
                conv_block(256, 512, 3, 1, 1)]],

            conv_block(512, 512, 1, 1, 0),
            conv_block(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            *[layer for _ in range(2) for layer in [
                conv_block(1024, 512, 1, 1, 0),
                conv_block(512, 1024, 3, 1, 1)]],

            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 2, 1),

            conv_block(1024, 1024, 3, 1, 1),
            conv_block(1024, 1024, 3, 1, 1)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


if __name__ == "__main__":
    ## TESTING ##
    model = YOLOv1()
    dummy_input = torch.randn(1, 3, 448, 448)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expected: (1, 7, 7, 30) for VOC
