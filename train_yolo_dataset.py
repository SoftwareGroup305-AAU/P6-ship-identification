import torch
import os
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import YOLOv8Dataset
from loss import SumSquaredErrorLoss
from models import YOLOv1, YOLOv1ResNet
import torch.optim.lr_scheduler as lrs

# Training configuration
BATCH_SIZE = 64
EPOCHS = 135
WARMUP_EPOCHS = 0
VAL_INTERVAL = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
GRID_SIZE = 7
NUM_PREDICTORS = 2
CHECKPOINT_INTERVAL = 20

# def lr_lambda(epoch):
#     if epoch < WARMUP_EPOCHS:
#         return (epoch + 1) / WARMUP_EPOCHS  
#     elif epoch < 100:
#         return 1.0
#     elif epoch < 130:
#         return 0.1
#     else:
#         return 0.01

if __name__ == '__main__':  # Prevent recursive subprocess creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)

    # Load the dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((448, 448))
    ])
    train_set = YOLOv8Dataset(r'data/ship-detection', 'train', grid_size=GRID_SIZE, num_predictors=NUM_PREDICTORS, transform=transform, normalize=True, augment=True)
    test_set = YOLOv8Dataset(r'data/ship-detection', 'val', grid_size=GRID_SIZE, num_predictors=NUM_PREDICTORS, transform=transform, normalize=True, augment=False)

    num_classes = train_set.C

    # Model and loss
    model = YOLOv1ResNet(num_classes, NUM_PREDICTORS).to(device)
    loss_function = SumSquaredErrorLoss(num_classes, NUM_PREDICTORS)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[id for id in range(torch.cuda.device_count())], output_device=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # scheduler = lrs.LambdaLR(optimizer, lr_lambda)


    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE
        
    )

    # Logging and checkpoint paths
    log_file_path = "log_resnet_ship.txt"
    weight_dir = "weights_resnet_ship"
    os.makedirs(weight_dir, exist_ok=True)

    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))

    def save_metrics():
        np.save('train_losses.npy', train_losses)
        np.save('test_losses.npy', test_losses)

    def log_to_file(message):
        with open(log_file_path, "a") as f:
            f.write(message + "\n")
        print(message)

    #####################
    #       Train       #
    #####################
    for epoch in tqdm(range(WARMUP_EPOCHS + EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc='Train', leave=False)
        for data, labels, _ in train_bar:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            train_bar.set_postfix(loss=loss.item())
            del data, labels
        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
        log_msg += f" | LR = {LEARNING_RATE:.6f}"
        # scheduler.step()
        if epoch % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            log_msg += f" | Test Loss = {test_loss:.4f}"
            # save_metrics()

        log_to_file(log_msg)

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(core_model.state_dict(), os.path.join(weight_dir, f'epoch_{epoch+1}.pth'))

    # save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final.pth'))
