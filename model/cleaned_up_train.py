import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, ops
from dataset import YOLODataset
from core import YOLO

import math

# Configurations
class Config:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training
    batch_size = 16
    epochs = 50
    initial_lr = 0.0001
    warmup_epochs = 5
    reg_max = 16
    
    # Data
    image_size = (640, 640)
    feat_size = (80, 80)  # image_size / 8
    num_classes = 11
    
    # Paths
    train_images = "data/train/images/"
    train_labels = "data/train/labels/"
    model_save_path = "yolo_custom.pth"

# Initialize
print(f"Using device: {Config.device}")
model = YOLO(Config.num_classes).to(Config.device)
gpu_count = torch.cuda.device_count()
print(f"Using {gpu_count} GPUs")

# Data Augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(Config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

# Dataset and DataLoader
train_data = YOLODataset(Config.train_images, Config.train_labels, train_transforms)
train_loader = DataLoader(train_data, batch_size=Config.batch_size, 
                         shuffle=True, collate_fn=collate_fn)

# Target Builder
def build_targets(targets_list, feat_size=Config.feat_size, reg_max=Config.reg_max, num_classes=Config.num_classes):
    B = len(targets_list)
    H, W = feat_size
    dfl_targets = torch.zeros((B, H, W, 4), dtype=torch.float32)
    ciou_targets = torch.zeros((B, H, W, 4), dtype=torch.float32)
    cls_targets = torch.zeros((B, H, W, num_classes), dtype=torch.float32)  # One-hot encoded
    
    for b, targets in enumerate(targets_list):
        for t in targets:
            if len(t) != 5: continue
            
            cls_idx, cx, cy, w, h = t  # Normalized [0,1]
            gx, gy = cx * W, cy * H
            gw, gh = w * W, h * H
            
            # Convert to corners
            x1, y1 = gx - gw/2, gy - gh/2
            x2, y2 = gx + gw/2, gy + gh/2
            gi, gj = int(gx), int(gy)

            if 0 <= gi < W and 0 <= gj < H:
                # DFL distances (asymmetric)
                l, t = gx - gi, gy - gj
                r, b = (gi + 1) - gx, (gj + 1) - gy
                
                # Clamp and store
                eps = 1e-4
                dfl_targets[int(b), gj, gi] = torch.clamp(
                    torch.tensor([l, t, r, b]), 0, reg_max - eps)
                
                # Normalized CIoU targets
                ciou_targets[int(b), gj, gi] = torch.tensor(
                    [x1/W, y1/H, x2/W, y2/H])
                
                # Class targets (one-hot)
                cls_targets[int(b), gj, gi, int(cls_idx)] = 1.0
    
    return (dfl_targets.to(Config.device), 
            ciou_targets.to(Config.device),
            cls_targets.to(Config.device))

def ciou_loss(preds, targets, scales=['p3', 'p5', 'p7'], reg_max=16):
    """Compute CIoU loss across scales using torchvision.ops.complete_box_iou_loss"""
    total_loss = 0.0
    num_levels = 0

    for scale in scales:
        bbox = preds[scale]['bbox']
        B, C, H, W = bbox.shape
        assert C == 4 * reg_max, f"Expected {4 * reg_max}, got {C}"

        bbox = bbox.view(B, 4, reg_max, H, W).softmax(dim=2)
        grid = torch.arange(reg_max, device=bbox.device, dtype=torch.float32)
        dist = (bbox * grid.view(1, 1, -1, 1, 1)).sum(2)  # [B, 4, H, W]
        dist = dist.permute(0, 2, 3, 1)  # [B, H, W, 4]

        cx = torch.arange(W, device=dist.device).float().view(1, 1, W) + 0.5
        cy = torch.arange(H, device=dist.device).float().view(1, H, 1) + 0.5
        cx = cx.expand(B, H, W)
        cy = cy.expand(B, H, W)

        x1 = cx - dist[..., 0]
        y1 = cy - dist[..., 1]
        x2 = cx + dist[..., 2]
        y2 = cy + dist[..., 3]
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

        t_resized = F.interpolate(targets.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False)
        t_resized = t_resized.permute(0, 2, 3, 1).contiguous().reshape(-1, 4)

        mask = (t_resized.sum(dim=1) != 0)

        if mask.any():
            loss = ops.complete_box_iou_loss(pred_boxes[mask], t_resized[mask], reduction='mean')
            total_loss += loss
            num_levels += 1

    if num_levels == 0:
        return torch.tensor(0.0, device=targets.device)
    
    return total_loss / num_levels

def focal_loss(preds, targets, alpha=0.25, gamma=2.0, scales=['p3', 'p5', 'p7']):
    """
    Focal loss using same target for all scales.
    preds: dict with keys like 'p3', 'p5', 'p7' each with 'cls' logits [B, C, H, W]
    targets: [B, H_t, W_t, C] one-hot class map (resized for each scale inside)
    """
    total_loss = 0.0
    num_levels = 0

    for scale in scales:
        cls_pred = preds[scale]['cls']  # [B, C, H, W]
        B, C, H, W = cls_pred.shape

        cls_target = F.interpolate(targets.permute(0, 3, 1, 2).float(), size=(H, W), mode='nearest')
        cls_target = cls_target.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(-1, C)      # [B*H*W, C]
        pred_prob = torch.sigmoid(cls_pred)

        ce_loss = -(cls_target * torch.log(pred_prob + 1e-8) + (1 - cls_target) * torch.log(1 - pred_prob + 1e-8))

        p_t = cls_target * pred_prob + (1 - cls_target) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        alpha_factor = cls_target * alpha + (1 - cls_target) * (1 - alpha)

        loss = modulating_factor * alpha_factor * ce_loss
        total_loss += loss.mean()
        num_levels += 1

    return total_loss / max(num_levels, 1)

def distributed_focal_loss(pred, target, reg_max=16, scales=['p3', 'p5', 'p7']):
    """
    pred: dict of model outputs per scale, each containing 'bbox' logits of shape [B, 4 * reg_max, H, W]
    target: float tensor of shape [B, H, W, 4], continuous values in [0, reg_max)
    """
    total_loss = 0.0
    num_levels = 0

    for scale in scales:
        bbox = pred[scale]['bbox']  # [B, 4*reg_max, H, W]
        B, C, H, W = bbox.shape
        assert C == 4 * reg_max, f"Expected {4 * reg_max} channels, got {C}"

        bbox = bbox.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2).contiguous()
        bbox = bbox.view(B, H * W, 4, reg_max)  # [B, HW, 4, reg_max]

        t_resized = F.interpolate(target.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False)
        t_resized = t_resized.permute(0, 2, 3, 1).contiguous().view(B, H * W, 4).float().to(bbox.device)

        for i in range(4):  # 4 box coordinates
            pred_dist = bbox[..., i, :]  # [B, HW, reg_max]
            t = t_resized[..., i]        # [B, HW]

            tl = t.long()
            tr = tl + 1
            wl = tr - t
            wr = 1 - wl

            tl = torch.clamp(tl, 0, reg_max - 1)
            tr = torch.clamp(tr, 0, reg_max - 1)
            pred_dist = torch.clamp(pred_dist, min=-30, max=30)

            loss_l = F.cross_entropy(pred_dist.view(-1, reg_max), tl.view(-1), reduction='none').view(B, H * W)
            loss_r = F.cross_entropy(pred_dist.view(-1, reg_max), tr.view(-1), reduction='none').view(B, H * W)
            dfl = loss_l * wl + loss_r * wr

            total_loss += dfl.mean()
        num_levels += 1

    return total_loss / (2 * num_levels)



# Training Setup
optimizer = optim.AdamW(model.parameters(), lr=Config.initial_lr)

# LR Scheduler with Warmup
warmup_scheduler = LinearLR(optimizer, 
                          start_factor=0.01,
                          end_factor=1.0,
                          total_iters=len(train_loader)*Config.warmup_epochs)

cosine_scheduler = CosineAnnealingLR(optimizer,
                                   T_max=len(train_loader)*(Config.epochs-Config.warmup_epochs))

# Combined Training Loop
def train(model, loader, optimizer, device):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(Config.epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            dfl_targets, ciou_targets, cls_targets = build_targets(targets)
            
            # Forward pass
            outputs = model(images)
            
            if (math.isnan(outputs['p3']['bbox'].mean())):
                print("")
            
            # Calculate losses
            dfl_loss = distributed_focal_loss(outputs, dfl_targets)
            bbox_loss = ciou_loss(outputs, ciou_targets)
            cls_loss = focal_loss(outputs, cls_targets)
            
            # Combined loss (you can adjust weights as needed)
            loss = dfl_loss + bbox_loss + cls_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # LR scheduling
            if epoch < Config.warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            if batch_idx % 10 == 9:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{Config.epochs} | "
                      f"Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {avg_loss:.4f} (DFL: {dfl_loss:.2f}, "
                      f"CIoU: {bbox_loss:.2f}, CLS: {cls_loss:.2f}) | "
                      f"LR: {lr:.2e}")
        
        # End of epoch
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

# Run Training
train(model, train_loader, optimizer, Config.device)