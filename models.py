import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, num_bboxes: int, num_classes: int):
        super().__init__()
        self.B = num_bboxes
        self.C = num_classes
        self.S = 7
        self.depth = self.B*5+self.C

        layers = [
            ### Conv 1 ###
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ### Conv 2 ###
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ### Conv 3 ###
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ### Conv 4 ###
            *[
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ] * 4,
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ### Conv 5 ###
            *[
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ] * 2,
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1),
            ### Conv 6 ###
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), 
            nn.LeakyReLU(negative_slope=0.1),
            ### Linear 1 ###
            nn.Flatten(),
            nn.Linear(self.S*self.S*1024, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            ### Linear 2 ###
            nn.Dropout(),
            nn.Linear(4096, self.S*self.S*self.depth),

        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.model(x)
        return torch.reshape(x, (x.shape[0], self.S, self.S, self.depth))
        
if __name__ == '__main__':
    ### TESTING ###
    model = YOLOv1(num_bboxes=2, num_classes=20)
    dummy_input = torch.randn(1, 3, 448, 448)
    out = model(dummy_input)
    print("Output shape:", out.shape) 