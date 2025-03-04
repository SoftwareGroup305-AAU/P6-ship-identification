import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms

def import_data():
    transform = transforms.Compose([
        transforms.Resize(32, 32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                            (0.5, 0.5, 0.5))
    ])

    training_set = 0 # training data location
    training_data_loader = dataloader(training_set, batch_size=32, shuffle=True)

    test_set = 0 # test data location
    test_data_loader = dataloader(test_set, batch_size=32, shuffle=False)


class CNN(nn.Module):
    # number of classes to identify: {freight, fishing, empty}
    # dont know if empty should be included in classes
    classes = 3

    # cnn definition
    def __init__(self, classes):
        super(CNN, self, classes).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fcl1 = nn.Linear(64, 128)
        self.fcl2 = nn.Linear(128, classes)

    # cnn pipeline definition
    def pipeline(self, image):
        image = self.pool(f.relu(self.conv1(image)))
        image = self.pool(f.relu(self.conv2(image)))
        image = f.relu(self.fcl1(image))
        image = self.fcl2(image)
