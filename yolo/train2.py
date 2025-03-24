import torch.optim as optim
import torch
from yolo2 import YOLO, YOLODataset
from utils import yolo_loss, generate_anchors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

training_images_dir = "yolo/data/train/images/"
training_labels_dir = "yolo/data/train/labels/"
grid_size = 7
num_classes= 11


anchor_config = ([0.1, 0.2, 0.4], [0.5, 1, 2])
anchors = generate_anchors(*anchor_config)

model = YOLO(grid_size=grid_size, num_classes=num_classes, num_anchors=len(anchors))
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = yolo_loss

training_data = YOLODataset(image_dir=training_images_dir, label_dir=training_labels_dir, grid_size=grid_size, num_classes=num_classes, anchors=anchors,  transform=train_transforms)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0
    model.train()
    for idx, data in enumerate(train_loader):
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if idx % 10== 9: 
            print(f"[{epoch+1}, {idx+1:5d}] loss: {running_loss / 10:.3f}")
            running_loss = 0

print("Finished training!")




