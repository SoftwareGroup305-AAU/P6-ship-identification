import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models import YOLO
from dataset import YOLOv8Dataset
from loss import CompositeLoss
import tqdm
from validation import validate_model
import utils
import os


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    BATCH_SIZE = 32
    NUM_CLASSES = 6
    LEARNING_RATE = 1e-4
    EPOCHS = 135
    STRIDE = 64
    VAL_INTERVAL = 10
    CHECKPOINT_INTERVAL = 20
    IMG_SIZE = (448, 448)
    LOG_FILE_PATH = "log.txt"
    WEIGHT_DIR = "weights"

def main():
    #################################
    #             SETUP             #
    #################################
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.LOG_FILE_PATH), exist_ok=True)
    model = YOLO(Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    if Config.NUM_GPUS > 1:
        model = nn.DataParallel(model, device_ids=[id for id in range(Config.NUM_GPUS)], output_device=0)

    criterion = CompositeLoss(Config.NUM_CLASSES)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )
    transform = T.Compose([
        T.Resize(Config.IMG_SIZE)
    ])

    train_set = YOLOv8Dataset("data/ship-detection_6", "train", grid_size=7, transform=transform)
    val_set = YOLOv8Dataset("data/ship-detection_6", "val", grid_size=7, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        drop_last=True
    )
    #################################
    #             TRAIN             #
    #################################
    for epoch in tqdm(range(Config.EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        train_prog_bar = tqdm(train_loader, desc='Train', leave=False)
        for data, targets in train_prog_bar:
            data = data.to(Config.DEVICE)
            targets = tuple(t.to(Config.DEVICE) for t in targets)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(*output, *targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
            train_prog_bar.set_postfix(loss=loss.item())
        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
        log_msg += f" | LR = {Config.LEARNING_RATE:.6f}"
        del data, targets

    #################################
    #              EVAL             #
    #################################
        if epoch % Config.VAL_INTERVAL == 0:
            result = validate_model(model, 
                                    val_loader, 
                                    Config.DEVICE, 
                                    iou_thresh=0.5, 
                                    num_classes=Config.NUM_CLASSES, 
                                    img_size=Config.IMG_SIZE[0], 
                                    stride=Config.STRIDE)
            mAP = result[0]
            log_msg += f" | Val mAP = {mAP:.4f}"
        utils.log_to_file(log_msg, Config.LOG_FILE_PATH)

    #################################
    #           CHECKPOINT          #
    #################################
        if (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0:
            core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(core_model.state_dict(), os.path.join(Config.WEIGHT_DIR, f'epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    main()
