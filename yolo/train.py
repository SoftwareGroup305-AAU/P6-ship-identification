import torch.optim as optim
import torch
from yolo import YOLO, YOLODataset, calculate_yolo_loss, generate_anchors, calculate_yolo_loss_v2, calculate_yolo_loss_v3
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
criterion = calculate_yolo_loss_v3

training_data = YOLODataset(image_dir=training_images_dir, label_dir=training_labels_dir, grid_size=grid_size, num_classes=num_classes, anchors=anchors,  transform=train_transforms)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

test_loss = 0;
num_epochs = 20
for epochs in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        test_loss = loss.item()
        print(test_loss)

print(test_loss)




