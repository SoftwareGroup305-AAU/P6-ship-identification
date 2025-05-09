import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import nms
from dataset import YOLODataset
from core import YOLO

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def decode_predictions(pred_loc, pred_cls,
                       conf_thresh=0.5,
                       stride=64,   # your network’s down-sampling factor
                       img_size=448):
    """
    pred_loc:  (B, 4, H, W)  — four normalized [0,1] offsets: (l, t, r, b)
    pred_cls:  (B, C, H, W) — raw logits
    """
    B, C, H, W = pred_cls.shape
    device = pred_cls.device

    # precompute grid centers
    shifts_x = (torch.arange(W, device=device) + 0.5) * stride
    shifts_y = (torch.arange(H, device=device) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    all_boxes, all_scores, all_labels = [], [], []
    for b in range(B):
        # 1) class scores
        cls_prob = torch.sigmoid(pred_cls[b]).view(C, -1)     # (C, H*W)
        scores, labels = cls_prob.max(dim=0)                  # best class per cell

        # 2) bbox offsets in pixels
        loc = pred_loc[b].view(4, -1)                         # (4, H*W)
        # assume l,t,r,b are normalized to [0,1] of the whole image:
        loc_px = loc * img_size                              # scale to pixels
        l, t, r, b_ = loc_px

        # 3) corner coordinates
        x1 = (grid_x - l).clamp(0, img_size)
        y1 = (grid_y - t).clamp(0, img_size)
        x2 = (grid_x + r).clamp(0, img_size)
        y2 = (grid_y + b_).clamp(0, img_size)

        # 4) filter by confidence + size
        keep = (scores > conf_thresh) & ((x2 - x1) > 1) & ((y2 - y1) > 1)
        idxs = keep.nonzero(as_tuple=False).squeeze(1)

        boxes = torch.stack([x1[idxs], y1[idxs], x2[idxs], y2[idxs]], dim=1)
        all_boxes.append(boxes)
        all_scores.append(scores[idxs])
        all_labels.append(labels[idxs])

    return all_boxes, all_scores, all_labels

def compute_map(stats, gt_counts, num_classes):
    aps = []
    log = []
    for cls in range(num_classes):
        cls_stats = sorted(stats[cls], key=lambda x: -x[0])
        if gt_counts.get(cls, 0) == 0:
            aps.append(0)
            log.append(f"Class {cls:2d}: No GT")
            continue
        tp = np.array([s[1] for s in cls_stats])
        fp = 1 - tp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / gt_counts[cls]
        precision = tp / (tp + fp + 1e-6)
        ap = 0
        for t in np.linspace(0, 1, 11):
            prec = precision[recall >= t].max() if np.any(recall >= t) else 0
            ap += prec / 11
        aps.append(ap)
        log.append(f"Class {cls:2d}: AP = {ap*100:5.2f}%")
    mAP = np.mean(aps)
    log.append(f"\n[mAP@0.5] = {mAP*100:5.2f}%")
    return mAP, "\n".join(log)

def validate_model(model, dataloader,
                   device='cpu', iou_thresh=0.5,
                   num_classes=6,        # match your train.py
                   img_size=448,
                   stride=64):
    model.to(device).eval()
    stats = defaultdict(list)
    stats_agn = []
    gt_counts = defaultdict(int)
    gt_count_total = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # your model returns (cls_out, obj_out, loc_out)
            cls_out, obj_out, loc_out = model(images)

            # permute to (B, C, H, W)
            pred_cls = cls_out.permute(0,3,1,2)
            pred_loc = loc_out.permute(0,3,1,2)

            # decode boxes & scores
            boxes_batch, scores_batch, labels_batch = decode_predictions(
                pred_loc, pred_cls,
                conf_thresh=0.5,
                stride=stride,
                img_size=img_size
            )

            for i in range(len(images)):
                boxes = boxes_batch[i]
                scores = scores_batch[i]
                labels = labels_batch[i]

                if len(boxes) == 0:
                    continue

                # NMS
                keep = nms(boxes, scores, iou_thresh)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                # build GT
                gt_boxes, gt_labels = [], []
                for cls_id, cx, cy, w, h in targets[i]:
                    x1 = (cx - w/2) * img_size
                    y1 = (cy - h/2) * img_size
                    x2 = (cx + w/2) * img_size
                    y2 = (cy + h/2) * img_size
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(int(cls_id))

                if not gt_boxes:
                    continue
                gt_boxes = torch.tensor(gt_boxes, device=device)
                gt_labels = torch.tensor(gt_labels, device=device)

                matched = set()
                for pb, ps, pc in zip(boxes, scores, labels):
                    # IoU w/ all GT
                    ious = ( 
                        (torch.min(pb[2], gt_boxes[:,2]) - torch.max(pb[0], gt_boxes[:,0]))
                        .clamp(0) *
                        (torch.min(pb[3], gt_boxes[:,3]) - torch.max(pb[1], gt_boxes[:,1]))
                        .clamp(0)
                    )
                    area1 = (pb[2]-pb[0])*(pb[3]-pb[1])
                    area2 = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
                    union = area1 + area2 - ious + 1e-6
                    iou_vals = ious/union

                    best_iou, best_idx = iou_vals.max(0)
                    is_tp = best_iou >= iou_thresh and best_idx.item() not in matched

                    stats[pc.item()].append((ps.item(), int(is_tp)))
                    stats_agn.append((ps.item(), int(best_iou>=iou_thresh)))

                    pred_cls_id = pc.item()
                    gt_cls_id   = gt_labels[best_idx].item()
                    conf_mat[gt_cls_id, pred_cls_id] += 1
                    if is_tp:
                        matched.add(best_idx.item())

                for cls in gt_labels.tolist():
                    gt_counts[cls] += 1
                    gt_count_total += 1

    mAP, log = compute_map(stats, gt_counts, num_classes)
    agn_stats = defaultdict(list)
    for score, tp in stats_agn:
        agn_stats[0].append((score, tp))
    map_agn, log_agn = compute_map(agn_stats, {0: gt_count_total}, 1)

    return mAP, log, map_agn, log_agn, conf_mat

def run_all_validations():
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 6
    img_size    = 448
    stride      = 64  # img_size / feature_map_size

    val_tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset    = YOLODataset("../data/valid/images", "../data/valid/labels", val_tfms)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True,
                            collate_fn=collate_fn)

    model = YOLO(num_classes=num_classes)
    # load your checkpoint here...
    ckpt = torch.load("initial_yolo_last.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    mAP, cls_log, map_agn, agn_log, conf_mat = validate_model(
        model, dataloader,
        device=device,
        iou_thresh=0.5,
        num_classes=num_classes,
        img_size=img_size,
        stride=stride
    )

    print(cls_log)
    print("\nClass-agnostic mAP@0.5:\n", agn_log)
    print("\nConfusion matrix:\n", conf_mat)

if __name__ == "__main__":
    run_all_validations()
