import torch.optim as optim
from yolo.yolo import YOLO, YOLODataset, calculate_yolo_loss, generate_anchors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

training_images_dir = "data/train/images/"
training_labels_dir = "data/train/labels/"
grid_size = 7
num_classes= 11


anchor_config = ([0.1, 0.2, 0.4], [0.5, 1, 2])
anchors = generate_anchors(*anchor_config)

model = YOLO(grid_size=grid_size, num_classes=num_classes, num_anchors=len(anchors))
model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = calculate_yolo_loss

training_data = YOLODataset(image_dir=training_images_dir, label_dir=training_labels_dir, grid_size=grid_size, num_classes=num_classes, anchors=anchors,  transform=train_transforms)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

num_epochs = 20
for epochs in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = images.to('cuda')
        targets = targets.to('cuda')

        predictions = model(images)

        loss = criterion(predictions, targets, num_classes=num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()




