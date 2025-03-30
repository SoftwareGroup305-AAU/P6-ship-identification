import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utilities import CompositeLoss
from dataset import YOLODataset
from core import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

training_images_dir = "yolo/data/train/images/"
training_labels_dir = "yolo/data/train/labels/"

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

training_data = YOLODataset(training_images_dir, training_labels_dir, train_transforms)
dataloader = DataLoader(training_data, batch_size=16, shuffle=True, collate_fn=collate_fn) # trying smaller batch size, should be better and less resource intensive according to some paper

num_classes = 11 # should be 2 when we get the proper data set
model = YOLO(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CompositeLoss()

epochs = 20

def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    model.to(device)

    inc = 0
    run_loss = 0
    avg_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            if (inc >= 150):
                break

            images = images.to(device)
            targets = [target.to(device) for target in targets]
            optimizer.zero_grad()
            class_predictions, objectness_predictions, localization_predictions = model(images)
            loss = criterion(class_predictions, objectness_predictions, localization_predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            run_loss = loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
            print(f"Run {inc+1}/150, Run Loss: {run_loss}")
            inc += 1
            avg_loss += run_loss
        print(f"Average Loss for Epoch: {avg_loss / inc}")
        avg_loss = 0 
        inc = 0

train(model, dataloader, optimizer, criterion, device, epochs)