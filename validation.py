import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import nms
from models import YOLO
from tqdm import tqdm

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def decode_predictions(loc_pred, cls_pred, obj_pred,
                       conf_thresh=0.5,
                       stride=64,
                       img_size=448):
    """
    loc_pred: (B, S, S, B, 4)
    cls_pred: (B, S, S, B, C)
    obj_pred: (B, S, S, B)
    """
    B, S, _, num_preds, _ = loc_pred.shape
    device = loc_pred.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=device), torch.arange(S, device=device), indexing="ij"
    )
    grid_x = grid_x.unsqueeze(-1).repeat(1, 1, num_preds)
    grid_y = grid_y.unsqueeze(-1).repeat(1, 1, num_preds)

    cx = (grid_x + 0.5) * stride
    cy = (grid_y + 0.5) * stride

    cx = cx.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1)
    cy = cy.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1)

    loc_pred = loc_pred.reshape(B, -1, 4)
    cls_pred = cls_pred.reshape(B, -1, cls_pred.shape[-1])
    obj_pred = obj_pred.reshape(B, -1)

    boxes_out, scores_out, labels_out = [], [], []
    for i in range(B):
        box = loc_pred[i]
        obj = torch.sigmoid(obj_pred[i])
        cls = torch.sigmoid(cls_pred[i])
        score, label = cls.max(dim=-1)
        score = score * obj

        keep = score > conf_thresh
        if keep.sum() == 0:
            boxes_out.append(torch.empty((0, 4), device=device))
            scores_out.append(torch.empty((0,), device=device))
            labels_out.append(torch.empty((0,), dtype=torch.int64, device=device))
            continue

        box = box[keep]
        score = score[keep]
        label = label[keep]
        x_ctr = cx[i][keep]
        y_ctr = cy[i][keep]

        x1 = (x_ctr - box[:, 2] * img_size / 2).clamp(0, img_size)
        y1 = (y_ctr - box[:, 3] * img_size / 2).clamp(0, img_size)
        x2 = (x_ctr + box[:, 2] * img_size / 2).clamp(0, img_size)
        y2 = (y_ctr + box[:, 3] * img_size / 2).clamp(0, img_size)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        boxes_out.append(boxes)
        scores_out.append(score)
        labels_out.append(label)

    return boxes_out, scores_out, labels_out


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
    model.eval()
    stats = defaultdict(list)
    stats_agn = []
    gt_counts = defaultdict(int)
    gt_count_total = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Val"):
            images = images.to(device)
            # your model returns (cls_out, obj_out, loc_out)
            cls_out, obj_out, loc_out = model(images)

            # permute to (B, C, H, W)
            # pred_cls = cls_out.permute(0,3,1,2)
            # pred_loc = loc_out.permute(0,3,1,2)

            # decode boxes & scores
            boxes_batch, scores_batch, labels_batch = decode_predictions(
                loc_out, cls_out, obj_out,
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