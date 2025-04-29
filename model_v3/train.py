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
lambda_ciou = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def make_cls_targets(targets_list, feat_size, stride, num_classes):
    B,H,W = len(targets_list), *feat_size
    cls_t = torch.zeros((B,num_classes,H,W),dtype=torch.float32)
    for b, t in enumerate(targets_list):
        for cls_id, cx,cy,_,_ in t:
            gi, gj = int(cx*W), int(cy*H)
            if 0<=gi<W and 0<=gj<H:
                cls_t[b, int(cls_id), gj, gi] = 1.0
    return cls_t


def build_dfl_targets(targets_list, feat_size, stride, reg_max=16):
    B = len(targets_list)
    H, W = feat_size
    bins = reg_max + 1

    dfl_targets  = torch.zeros((B, H, W, 4), dtype=torch.float32)
    ciou_targets = torch.zeros((B, H, W, 4), dtype=torch.float32)

    img_w, img_h = W * stride, H * stride
    eps = 1e-4
    max_val = reg_max + eps

    for bi, tlist in enumerate(targets_list):
        for cls_id, cx, cy, w_n, h_n in tlist:
            px_cx, px_cy = cx * img_w, cy * img_h
            px_w, px_h   = w_n * img_w, h_n * img_h

            # GT corners
            x1 = px_cx - px_w/2
            y1 = px_cy - px_h/2
            x2 = px_cx + px_w/2
            y2 = px_cy + px_h/2

            # which cell?
            gx, gy = px_cx / stride, px_cy / stride
            gi, gj = int(gx), int(gy)
            if not (0 <= gi < W and 0 <= gj < H):
                continue

            # distances in cell‐units
            left   = gx - (x1/stride)
            top    = gy - (y1/stride)
            right  = (x2/stride) - gx
            bottom = (y2/stride) - gy

            # clamp into [0, reg_max]
            left   = float(min(max(left,   0.0), max_val))
            top    = float(min(max(top,    0.0), max_val))
            right  = float(min(max(right,  0.0), max_val))
            bottom = float(min(max(bottom, 0.0), max_val))

            dfl_targets [bi, gj, gi] = torch.tensor([left, top, right, bottom])
            ciou_targets[bi, gj, gi] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    return dfl_targets, ciou_targets



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
model = YOLO(
    num_classes=num_classes,
    depth_multiple=0.33,
    width_multiple=0.25,
    max_channels=1024
)

model.to(device)
gpu_count = torch.cuda.device_count()
# if gpu_count > 1:
#     model = nn.DataParallel(model, device_ids=[id for id in range(gpu_count)], output_device=0)

#↑↑↑ cant get multi-gpu to work for now↑↑↑

#def ciou_loss(preds, ciou_targets):
#    bbox = preds['p3']['bbox']

from torchvision.ops import complete_box_iou_loss

def ciou_loss(preds, ciou_targets, stride, reg_max=16):
    bbox = preds['p3']['bbox']
    B, C, H, W = bbox.shape
    bins = C // 4

    # distribution → distances
    p = bbox.view(B, 4, bins, H, W)
    p = F.softmax(p, dim=2)
    idx = torch.arange(bins, device=bbox.device, dtype=torch.float32)
    dist = (p * idx.view(1,1,-1,1,1)).sum(dim=2)       # [B,4,H,W]
    dist = dist.permute(0,2,3,1)                        # [B,H,W,4]
    l, t, r, b_ = dist.unbind(-1)

    # grid centers in pixel space
    shifts_x = (torch.arange(W, device=bbox.device) + 0.5) * stride
    shifts_y = (torch.arange(H, device=bbox.device) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    centers = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)

    # to (x1,y1,x2,y2)
    x1 = centers[...,0] - l * stride
    y1 = centers[...,1] - t * stride
    x2 = centers[...,0] + r * stride
    y2 = centers[...,1] + b_ * stride

    pred_boxes   = torch.stack([x1,y1,x2,y2], dim=-1).reshape(-1,4)
    target_boxes = ciou_targets.reshape(-1,4).to(bbox.device)

    mask = (target_boxes.sum(dim=1) != 0)
    if mask.any():
        return complete_box_iou_loss(pred_boxes[mask], target_boxes[mask], reduction='mean')
    else:
        return torch.tensor(0.0, device=bbox.device)

def distributed_focal_loss(pred, targets, reg_max=16):
    total_loss, num_scales = 0.0, 0
    for scale, pred_dict in pred.items():
        if scale not in targets:
            continue
        bbox_pred   = pred_dict['bbox']
        bbox_target = targets[scale].to(bbox_pred.device)

        B, C, H, W = bbox_pred.shape
        bins = reg_max + 1
        assert C == 4 * bins, f"Expected {4*bins}, got {C}"

        # [B, H, W, 4, bins]
        p = bbox_pred.view(B, 4, bins, H, W).permute(0, 3, 4, 1, 2)
        scale_loss = 0.0

        for i in range(4):
            pred_dist = p[..., i, :]  # [B, H, W, bins]
            t         = bbox_target[..., i]  # [B, H, W]

            # integer bin indices
            tl = t.long().clamp(0, bins - 1)
            tr = (tl + 1).clamp(0, bins - 1)
            wl = (tr.float() - t)
            wr = (1.0 - wl)

            # use reshape instead of view to handle non-contiguous
            loss_l = F.cross_entropy(
                pred_dist.reshape(-1, bins),
                tl.reshape(-1),
                reduction='none'
            ).reshape(t.shape)
            loss_r = F.cross_entropy(
                pred_dist.reshape(-1, bins),
                tr.reshape(-1),
                reduction='none'
            ).reshape(t.shape)

            scale_loss += (loss_l * wl + loss_r * wr).mean()

        total_loss += scale_loss * 0.5
        num_scales += 1

    return total_loss / num_scales if num_scales > 0 else torch.tensor(0.0, device=bbox_pred.device)


print(f"Using { 1 if gpu_count >= 1 else 0} GPUs")
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = distributed_focal_loss

epochs = 10


def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        for idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)

            # 1) Build DFL & CIoU targets
            dfl_p3, cio_t3 = build_dfl_targets(targets, (80,80), stride=8)
            dfl_p5, cio_t5 = build_dfl_targets(targets, (40,40), stride=16)
            dfl_p7, cio_t7 = build_dfl_targets(targets, (20,20), stride=32)
            dfl_targets    = {'p3': dfl_p3, 'p5': dfl_p5, 'p7': dfl_p7}

            # 2) Build classification targets
            cls_t3 = make_cls_targets(targets, (80,80), 8,  num_classes).to(device)
            cls_t5 = make_cls_targets(targets, (40,40),16, num_classes).to(device)
            cls_t7 = make_cls_targets(targets, (20,20),32, num_classes).to(device)

            optimizer.zero_grad()
            
            # ── You must get model outputs before using `output` ──────────────
            output = model(images)

            # 3) DFL loss
            dfl_loss = criterion(output, dfl_targets, reg_max=16)

            # 4) Classification loss
            cls_loss = (
                F.binary_cross_entropy_with_logits(output['p3']['cls'], cls_t3) +
                F.binary_cross_entropy_with_logits(output['p5']['cls'], cls_t5) +
                F.binary_cross_entropy_with_logits(output['p7']['cls'], cls_t7)
            )

            # 5) CIoU loss (sum over scales)
            ciou_l = 0.0
            ciou_l += ciou_loss({'p3': output['p3']}, cio_t3.to(device), stride=8)
            ciou_l += ciou_loss({'p3': output['p5']}, cio_t5.to(device), stride=16)
            ciou_l += ciou_loss({'p3': output['p7']}, cio_t7.to(device), stride=32)

            # 6) Total loss
            loss = dfl_loss + cls_loss + lambda_ciou * ciou_l
            loss.backward()
            optimizer.step()

            if idx % 10 == 9:
                print(f"[Epoch {epoch+1}, Batch {idx+1}] Loss: {loss.item():.4f}")
                run_loss = 0

        print(f"Epoch {epoch+1} complete. Last batch loss: {loss.item():.4f}")


        epoch_loss_avg = epoch_loss / inc

        #print(f"Average Loss for Epoch: {avg_loss}")

        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom_last.pth")
        if (epoch_loss_avg < best_loss):
            print(f"Average Loss for Epoch: {epoch_loss_avg:.4f}")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom_best.pth")
            best_loss = epoch_loss_avg

train(model, dataloader, optimizer, criterion, device, epochs)

torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "yolo_custom.pth")