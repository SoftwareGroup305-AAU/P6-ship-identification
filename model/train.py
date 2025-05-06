import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utilities import CompositeLoss
from dataset import YOLODataset
from core import YOLO

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_IMAGES_DIR = "data/train/images"
    TRAIN_LABELS_DIR = "data/train/labels"
    BATCH_SIZE = 32
    NUM_CLASSES = 6
    NUM_BBOXES = 2
    INITIAL_LEARNING_RATE = 0.00001
    EPOCHS = 135
    SAVE_PATH_BEST = "initial_yolo_best.pth"
    SAVE_PATH_LAST = "initial_yolo_last.pth"
    SAVE_PATH_FINAL = "initial_yolo.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.RandomCrop(448, padding=4),
    transforms.ToTensor()
])

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

training_data = YOLODataset(Config.TRAIN_IMAGES_DIR, Config.TRAIN_LABELS_DIR, train_transforms)
dataloader = DataLoader(training_data, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn) 

model = YOLO(Config.NUM_CLASSES)
model.to(device)
gpu_count = torch.cuda.device_count()

print(f"Using { 1 if gpu_count >= 1 else 0} GPUs")
optimizer = optim.Adam(model.parameters(), lr=Config.INITIAL_LEARNING_RATE)
criterion = CompositeLoss(Config.NUM_CLASSES)

epochs = 50

def train(model, dataloader, optimizer, criterion, device, epochs):
    best_loss = float('inf')
    model.train()
    
    for epoch in range(Config.EPOCHS):
        running_loss = 0.0
        epoch_loss = 0.0
        lr = Config.INITIAL_LEARNING_RATE
        
        if epoch == 1:
            lr = lr
        elif epoch > 1 and epoch <= 5:
            lr += 0.00002
        elif epoch > 5 and epoch <= 40:
            lr = 0.0001
        elif epoch > 40 and epoch <= 80:
            lr = 0.00001
        else:
            lr = 0.000001

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch_idx, data in enumerate(dataloader):
            images, targets = data

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            optimizer.zero_grad()

            class_predictions, objectness_predictions, localization_predictions = model(images)
            loss = criterion(class_predictions, objectness_predictions, localization_predictions, targets, device)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch+1}/{Config.EPOCHS}] "
                      f"Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {avg_loss:.3f}")
                running_loss = 0.0
        
        # Epoch summary
        epoch_avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] "
              f"Average Loss: {epoch_avg_loss:.3f} "
              f"Learning Rate: {lr}")
        
        # Save checkpoints
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_avg_loss,
        }, Config.SAVE_PATH_LAST)

        
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, Config.SAVE_PATH_BEST)

train(model, dataloader, optimizer, criterion, device, epochs)

torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom.pth")