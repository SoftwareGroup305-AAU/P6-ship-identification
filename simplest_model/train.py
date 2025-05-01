import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utilities import generate_targets, yolo_loss
from dataset import YOLODataset
from core import YOLOv1
from reduced_core import YOLOv1Reduced

# Configuration
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_IMAGES_DIR = "data/train/images"
    TRAIN_LABELS_DIR = "data/train/labels"
    BATCH_SIZE = 16
    NUM_CLASSES = 6
    NUM_BBOXES = 2
    INITIAL_LEARNING_RATE = 0.00001
    EPOCHS = 50
    SAVE_PATH_BEST = "reduced_yolo_custom_best.pth"
    SAVE_PATH_LAST = "reduced_yolo_custom_last.pth"
    SAVE_PATH_FINAL = "reduced_yolo_custom.pth"

def create_dataloader():
    """Create and return the training DataLoader"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images), list(targets)

    dataset = YOLODataset(
        Config.TRAIN_IMAGES_DIR,
        Config.TRAIN_LABELS_DIR,
        transform
    )
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

def setup_model():
    """Initialize model and optimizer"""
    model = YOLOv1Reduced(
        number_of_bboxes=Config.NUM_BBOXES,
        number_of_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    # Multi-GPU support (commented out as per your note)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.INITIAL_LEARNING_RATE)
    return model, optimizer

def train(model, dataloader, optimizer, device):
    """Training loop"""
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

        for batch_idx, (images, raw_targets) in enumerate(dataloader, 1):
            print("Batch: ", batch_idx)
            # Move data to device
            images = images.to(device)
            raw_targets = [target.to(device) for target in raw_targets]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            targets = generate_targets(outputs, raw_targets, Config.NUM_BBOXES)
            loss = yolo_loss(outputs, targets, Config.NUM_BBOXES)

            if loss.item() > 30:
                print(f"Spike detected: {loss.item()}")
                print(f"Targets: {raw_targets}")
                print(f"Predictions: {outputs}")

            
            # Backward pass
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
            else:
                print("nan detected, loss skipped:", loss.item())

            # Logging
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
        torch.save(state_dict, Config.SAVE_PATH_LAST)
        
        if epoch_avg_loss < best_loss:
            torch.save(state_dict, Config.SAVE_PATH_BEST)
            best_loss = epoch_avg_loss

def main():
    print(f"Using device: {Config.DEVICE}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    dataloader = create_dataloader()
    model, optimizer = setup_model()
    
    train(model, dataloader, optimizer, Config.DEVICE)
    
    # Save final model
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, Config.SAVE_PATH_FINAL)

if __name__ == "__main__":
    main()