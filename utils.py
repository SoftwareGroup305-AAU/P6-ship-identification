import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF

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

def plot_boxes(image_tensor, class_target, objectness_target, localization_target, class_names=None, conf_threshold=0.5, max_overlap=0.5):
    """
    Visualizes YOLO-style grid-based targets on an image tensor, with optional overlap suppression.
    """
    image = TF.to_pil_image(image_tensor.cpu())
    w, h = image.size
    S = objectness_target.shape[0]

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    boxes = []

    # Collect all boxes above confidence threshold
    for i in range(S):
        for j in range(S):
            obj_score = objectness_target[i, j].item()
            if obj_score < conf_threshold:
                continue

            x, y, bw, bh = localization_target[i, j]
            x *= w
            y *= h
            bw *= w
            bh *= h
            x1 = x - bw / 2
            y1 = y - bh / 2
            x2 = x + bw / 2
            y2 = y + bh / 2

            class_idx = class_target[i, j].argmax().item() if class_names else -1
            boxes.append({
                'coords': [x1, y1, x2, y2],
                'conf': obj_score,
                'class_idx': class_idx
            })

    # Sort by confidence
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
