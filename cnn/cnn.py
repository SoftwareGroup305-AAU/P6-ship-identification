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