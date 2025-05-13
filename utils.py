import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
import random
import math

class YoloAugment:
    def __init__(self, resize = None, hflip_prob=0, rotate_prob=0, max_rotate_angle=30, augment=False):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.rotate_prob = rotate_prob
        self.max_rotate_angle = max_rotate_angle
        self.augment = augment

    def __call__(self, image, labels):
        if self.resize:
            image = TF.resize(image, self.resize)
        _, h, w = image.shape

        # Random rotation by arbitrary angle
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotate_angle, self.max_rotate_angle)
            image = TF.rotate(image, angle, expand=False)
            angle_rad = -math.radians(angle)  # Negative for correct direction

            # For each label, rotate the box
            for label in labels:
                x, y, bw, bh = label[1], label[2], label[3], label[4]
                # Convert normalized to pixel
                cx = x * w
                cy = y * h
                bw_pix = bw * w
                bh_pix = bh * h

                # Get box corners
                corners = [
                    [cx - bw_pix/2, cy - bh_pix/2],
                    [cx + bw_pix/2, cy - bh_pix/2],
                    [cx + bw_pix/2, cy + bh_pix/2],
                    [cx - bw_pix/2, cy + bh_pix/2],
                ]
                # Rotate corners
                new_corners = []
                ox, oy = w/2, h/2  # image center
                for px, py in corners:
                    dx, dy = px - ox, py - oy
                    qx = ox + math.cos(angle_rad) * dx - math.sin(angle_rad) * dy
                    qy = oy + math.sin(angle_rad) * dx + math.cos(angle_rad) * dy
                    new_corners.append([qx, qy])
                xs = [pt[0] for pt in new_corners]
                ys = [pt[1] for pt in new_corners]
                # Get new box
                min_x, max_x = max(0, min(xs)), min(w, max(xs))
                min_y, max_y = max(0, min(ys)), min(h, max(ys))
                # Convert back to YOLO normalized
                new_cx = (min_x + max_x) / 2 / w
                new_cy = (min_y + max_y) / 2 / h
                new_bw = (max_x - min_x) / w
                new_bh = (max_y - min_y) / h
                label[1], label[2], label[3], label[4] = new_cx, new_cy, new_bw, new_bh

        # Random horizontal flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            for label in labels:
                label[1] = 1.0 - label[1]  # Flip x_center
        
        if self.augment:
            image = TF.adjust_saturation(image, 0.2 * random.random() + 0.9)
            image = TF.adjust_hue(image, 0.2 * random.random() - 0.1)
        return image, labels


def compute_iou(box1, box2):
    """Computes IoU between two boxes: [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def plot_boxes(image_tensor, class_target, objectness_target, localization_target,
               class_names=None, conf_threshold=0.5, max_overlap=0.5):
    """
    Visualizes YOLO-style grid-based targets with multiple predictors per cell.
    """
    image = TF.to_pil_image(image_tensor.cpu())
    w, h = image.size
    S, _, B = objectness_target.shape

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    boxes = []

    for i in range(S):
        for j in range(S):
            for b in range(B):
                obj_score = objectness_target[i, j, b].item()
                if obj_score < conf_threshold:
                    continue

                x, y, bw, bh = localization_target[i, j, b]
                x *= w
                y *= h
                bw *= w
                bh *= h
                x1 = x - bw / 2
                y1 = y - bh / 2
                x2 = x + bw / 2
                y2 = y + bh / 2

                if class_names:
                    class_idx = class_target[i, j, b].argmax().item()
                else:
                    class_idx = -1

                boxes.append({
                    'coords': [x1, y1, x2, y2],
                    'conf': obj_score,
                    'class_idx': class_idx
                })

    # Sort and apply NMS-like overlap suppression
    boxes.sort(key=lambda b: b['conf'], reverse=True)
    final_boxes = []

    while boxes:
        best = boxes.pop(0)
        final_boxes.append(best)
        boxes = [b for b in boxes if compute_iou(best['coords'], b['coords']) < max_overlap]

    # Draw remaining boxes
    for box in final_boxes:
        x1, y1, x2, y2 = box['coords']
        bw = x2 - x1
        bh = y2 - y1
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        if class_names:
            label = class_names[box['class_idx']]
            confidence = box['conf'] * 100
            text = f"{label} {confidence:.1f}%"
            ax.text(x1, y1 - 5, text, color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def log_to_file(message, file_path):
    with open(file_path, "a") as f:
        f.write(message + "\n")
    print(message)
