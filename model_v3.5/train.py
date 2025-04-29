import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from dataset import YOLODataset
from core import YOLO
#from utils import build_dfl_targets
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

def build_multi_scale_dfl_targets(targets_list, feat_sizes={'p3': (320, 320), 'p5': (160, 160), 'p7': (40, 40)}, reg_max=16):
    B = len(targets_list)
    dfl_targets = {}

    for scale_name, (H, W) in feat_sizes.items():
        scale_target = torch.zeros((B, H, W, 4), dtype=torch.float32)

        for b, targets in enumerate(targets_list):
            for t in targets:
                if len(t) != 5:
                    continue
                _, cx, cy, w, h = t  # normalized

                gx = cx * W
                gy = cy * H
                gw = w * W
                gh = h * H

                x1 = gx - gw / 2
                y1 = gy - gh / 2
                x2 = gx + gw / 2
                y2 = gy + gh / 2

                gi = int(gx)
                gj = int(gy)

                if 0 <= gi < W and 0 <= gj < H:
                    eps = 1e-4
                    l = gx - x1
                    t_ = gy - y1
                    r = x2 - gx
                    b_ = y2 - gy

                    l = l / (W / (reg_max - 1))
                    t_ = t_ / (H / (reg_max - 1))
                    r = r / (W / (reg_max - 1))
                    b_ = b_ / (H / (reg_max - 1))

                    scale_target[b, gj, gi] = torch.tensor([
                        min(max(l, 0), reg_max - eps),
                        min(max(t_, 0), reg_max - eps),
                        min(max(r, 0), reg_max - eps),
                        min(max(b_, 0), reg_max - eps)
                    ], dtype=torch.float32)

        dfl_targets[scale_name] = scale_target

    return dfl_targets





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
dataloader = DataLoader(training_data, batch_size=8, shuffle=True, collate_fn=collate_fn) # trying smaller batch size, should be better and less resource intensive according to some paper

num_classes = 11 # should be 2 when we get the proper data set
model = YOLO(num_classes)
model.to(device)
gpu_count = torch.cuda.device_count()
# if gpu_count > 1:
#     model = nn.DataParallel(model, device_ids=[id for id in range(gpu_count)], output_device=0)

#↑↑↑ cant get multi-gpu to work for now↑↑↑

def ciou_loss(preds, ciou_targets):
    bbox = preds['p3']['bbox']

from torchvision.ops import complete_box_iou_loss

def ciou_loss(preds, ciou_targets):
    """
    preds: model output dict with 'p3' containing 'bbox' logits of shape [B, 4 * reg_max, H, W]
    ciou_targets: float tensor of shape [B, H, W, 4] containing (x1, y1, x2, y2) format boxes
    """
    # Get predicted bounding boxes (in DFL format)
    bbox = preds['p3']['bbox']
    B, C, H, W = bbox.shape
    reg_max = C // 4
    
    # Convert DFL predictions to box coordinates
    # First, reshape and softmax the distribution
    bbox = bbox.view(B, 4, reg_max, H, W)
    bbox = F.softmax(bbox, dim=2)
    
    # Create the grid of possible values (0 to reg_max-1)
    grid = torch.arange(reg_max, dtype=torch.float, device=bbox.device)
    
    # Calculate expected value (integral over the distribution)
    pred_dist = (bbox * grid.view(1, 1, -1, 1, 1)).sum(dim=2)  # [B, 4, H, W]
    
    # Convert distances to box coordinates (x1, y1, x2, y2)
    pred_dist = pred_dist.permute(0, 2, 3, 1)  # [B, H, W, 4]
    pred_ltrb = pred_dist  # left, top, right, bottom distances
    
    # Convert to (x1, y1, x2, y2) format
    center_x = torch.arange(W, device=bbox.device).view(1, 1, W) + 0.5  # grid centers
    center_y = torch.arange(H, device=bbox.device).view(1, H, 1) + 0.5
    
    pred_boxes = torch.zeros_like(pred_ltrb)
    pred_boxes[..., 0] = center_x - pred_ltrb[..., 0]  # x1 = center_x - left
    pred_boxes[..., 1] = center_y - pred_ltrb[..., 1]  # y1 = center_y - top
    pred_boxes[..., 2] = center_x + pred_ltrb[..., 2]  # x2 = center_x + right
    pred_boxes[..., 3] = center_y + pred_ltrb[..., 3]  # y2 = center_y + bottom
    
    # Get target boxes (already in x1,y1,x2,y2 format)
    target_boxes = ciou_targets.to(bbox.device)
    
    # Reshape boxes for torchvision ops
    pred_boxes = pred_boxes.reshape(-1, 4)
    target_boxes = target_boxes.reshape(-1, 4)
    
    # Only compute loss where there are targets (ciou_targets != 0)
    mask = (target_boxes.sum(dim=1) != 0)
    
    if mask.any():
        return complete_box_iou_loss(
            pred_boxes[mask],
            target_boxes[mask],
            reduction='mean'
        )
    else:
        return torch.tensor(0.0, device=pred_boxes.device)

def distributed_focal_loss(pred, targets, reg_max=16):
    total_loss = 0.0
    num_scales = 0

    for scale_name in ['p3', 'p5', 'p7']:
        if scale_name not in pred or scale_name not in targets:
            continue

        pred_dict = pred[scale_name]
        bbox_pred = pred_dict['bbox']
        bbox_target = targets[scale_name].to(bbox_pred.device).float()

        B, C, H, W = bbox_pred.shape
        assert C == 4 * reg_max, f"Expected {4 * reg_max} channels, got {C}"

        bbox_pred = bbox_pred.view(B, 4, reg_max, H, W)
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, 4, reg_max)
        bbox_target = bbox_target  # (B, H, W, 4)

        pred_dist = bbox_pred.reshape(-1, 4, reg_max)  # (B*H*W, 4, reg_max)
        t = bbox_target.reshape(-1, 4)                 # (B*H*W, 4)

        scale_loss = 0.0

        for i in range(4):
            pred_i = pred_dist[:, i, :]  # (B*H*W, reg_max)
            t_i = t[:, i]                # (B*H*W,)

            t_i = torch.clamp(t_i, 0, reg_max - 1 - 1e-6)

            left_bin = t_i.floor().long()
            right_bin = left_bin + 1

            weight_right = t_i - left_bin.float()
            weight_left = 1.0 - weight_right

            right_bin = torch.clamp(right_bin, 0, reg_max - 1)
            left_bin = torch.clamp(left_bin, 0, reg_max - 1)

            loss_left = F.cross_entropy(pred_i, left_bin, reduction='none')
            loss_right = F.cross_entropy(pred_i, right_bin, reduction='none')

            loss = weight_left * loss_left + weight_right * loss_right

            scale_loss += loss.mean()

        total_loss += scale_loss / 2  # average ltrb
        num_scales += 1

    return total_loss / num_scales if num_scales > 0 else 0.0
        

print(f"Using { 1 if gpu_count >= 1 else 0} GPUs")
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = distributed_focal_loss

epochs = 10


def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    best_loss = float("inf")
    inc = 0
    run_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, data in enumerate(dataloader):
            images, targets = data
            # if (inc >= 150):
            #     break
            images = images.to(device)
            #dfl_targets, ciou_targets = build_dfl_targets(targets, feat_size=(80, 80), reg_max=16)
            dfl_targets = build_multi_scale_dfl_targets(targets, reg_max=16)

            optimizer.zero_grad()
            output = model(images)
            #ciou_loss_v = ciou_loss(output, ciou_targets)
            dfl_loss = criterion(output, dfl_targets)
            loss = dfl_loss #+ ciou_loss_v
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            epoch_loss += loss_val
            run_loss += loss_val
            inc += 1

            if idx % 10 == 9:
                print(f"[Epoch {epoch+1}, Batch {idx+1}] Loss: {run_loss / 10:.4f}")
                run_loss = 0

            epoch_loss_avg = epoch_loss / inc

        #print(f"Average Loss for Epoch: {avg_loss}")

        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom_last.pth")
        if (epoch_loss_avg < best_loss):
            print(f"Average Loss for Epoch: {epoch_loss_avg:.4f}")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom_best.pth")
            best_loss = epoch_loss_avg

train(model, dataloader, optimizer, criterion, device, epochs)

torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom.pth")