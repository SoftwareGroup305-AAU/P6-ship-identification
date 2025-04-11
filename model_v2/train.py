import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import YOLODataset
from core import YOLO
from utilities import YOLOLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

training_images_dir = "yolo/data/train/images/"
training_labels_dir = "yolo/data/train/labels/"

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
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
model.to(device)
gpu_count = torch.cuda.device_count()
# if gpu_count > 1:
#     model = nn.DataParallel(model, device_ids=[id for id in range(gpu_count)], output_device=0)

#↑↑↑ cant get multi-gpu to work for now↑↑↑

print(f"Using { 1 if gpu_count >= 1 else 0} GPUs")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = YOLOLoss(num_classes)

epochs = 50

def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    best_loss = float("inf")
    inc = 0
    run_loss = 0
    avg_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, data in enumerate(dataloader):
            images, targets = data
            # if (inc >= 150):
            #     break
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, None)
            # loss.backward()
            # optimizer.step()
            # epoch_loss += loss.item()
            # run_loss += loss.item()
            if idx % 10== 9: 
                print(f"[{epoch+1}, {idx+1:5d}] loss: {run_loss / 10:.3f}")
                run_loss = 0
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
            # print(f"Run {inc+1}/150, Run Loss: {run_loss}")
            inc += 1
            avg_loss += run_loss
        epoch_loss_avg = epoch_loss / inc
        print(f"Average Loss for Epoch: {epoch_loss_avg}")
        inc = 0
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom_last.pth")
        if (epoch_loss_avg < best_loss):
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom_best.pth")
            best_loss = epoch_loss_avg

train(model, dataloader, optimizer, criterion, device, epochs)

torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom.pth")