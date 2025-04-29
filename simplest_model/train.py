import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utilities import generate_targets, yolo_loss
from dataset import YOLODataset
from core import YOLOv1

# Configuration
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_IMAGES_DIR = "yolo/data/train/images/"
    TRAIN_LABELS_DIR = "yolo/data/train/labels/"
    BATCH_SIZE = 16
    NUM_CLASSES = 11  # Should be 2 for final dataset
    NUM_BBOXES = 2
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    SAVE_PATH_BEST = "yolo_custom_best.pth"
    SAVE_PATH_LAST = "yolo_custom_last.pth"
    SAVE_PATH_FINAL = "yolo_custom.pth"

def create_dataloader():
    """Create and return the training DataLoader"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
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
    model = YOLOv1(
        number_of_bboxes=Config.NUM_BBOXES,
        number_of_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    # Multi-GPU support (commented out as per your note)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    return model, optimizer

def train(model, dataloader, optimizer, device):
    """Training loop"""
    best_loss = float('inf')
    model.train()
    
    for epoch in range(Config.EPOCHS):
        running_loss = 0.0
        epoch_loss = 0.0
        
        for batch_idx, (images, raw_targets) in enumerate(dataloader, 1):
            # Move data to device
            images = images.to(device)
            raw_targets = [target.to(device) for target in raw_targets]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            targets = generate_targets(outputs, raw_targets, Config.NUM_BBOXES)
            loss = yolo_loss(outputs, targets, Config.NUM_BBOXES)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
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
              f"Average Loss: {epoch_avg_loss:.3f}")
        
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