from torchvision.io import read_image
from torchvision import transforms
from core import YOLO
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms

# Define transformations
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Paths
model_file = r"yolo_custom.pth"
test_img = r"sail_boat.jpg"

# Load and preprocess the image
img = read_image(test_img)
img = train_transforms(img).unsqueeze(0)

# Initialize model
model = YOLO(num_classes=11)
model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
model.eval()

# Forward pass - using p3, p5, and p7 scales
with torch.no_grad():
    predictions = model(img)
    p3_pred = predictions["p3"]
    p5_pred = predictions["p5"]
    p7_pred = predictions["p7"]

# Decode function for p3, p5, and p7 predictions
def decode_dfl_predictions(pred_bbox, pred_cls, reg_max=16, conf_thresh=0.12, stride=8, img_size=640):
    B, _, H, W = pred_bbox.shape
    device = pred_bbox.device

    # Decode distribution to box coordinates
    pred_bbox = pred_bbox.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2)
    pred_bbox = F.softmax(pred_bbox, dim=-1)
    pred_bbox = pred_bbox @ torch.arange(reg_max).float().to(device)

    boxes, scores, class_ids = [], [], []

    for b in range(B):
        for i in range(H):
            for j in range(W):
                l, t, r, b_ = pred_bbox[b, i, j]
                x_center = (j + 0.5) * stride
                y_center = (i + 0.5) * stride

                # Calculate box coordinates with clamping
                x_min = max(0, min(x_center - l * stride, img_size))
                y_min = max(0, min(y_center - t * stride, img_size))
                x_max = max(0, min(x_center + r * stride, img_size))
                y_max = max(0, min(y_center + b_ * stride, img_size))

                w = x_max - x_min
                h = y_max - y_min

                # Process class predictions
                class_logits = pred_cls[b, :, i, j]
                class_scores = torch.softmax(class_logits, dim=0)
                score, class_id = torch.max(class_scores, dim=0)

                # Filter predictions
                if score.item() > conf_thresh and w > 1 and h > 1:
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append(score.item())
                    class_ids.append(class_id.item())

    return boxes, scores, class_ids

# Decode p3, p5, p7 predictions
def decode_all_predictions(p3_pred, p5_pred, p7_pred):
    # Apply the decode function to p3, p5, and p7 with their respective strides
    p3_boxes, p3_scores, p3_class_ids = decode_dfl_predictions(
        p3_pred["bbox"], 
        p3_pred["cls"], 
        stride=8  # p3 has stride 8
    )

    p5_boxes, p5_scores, p5_class_ids = decode_dfl_predictions(
        p5_pred["bbox"], 
        p5_pred["cls"], 
        stride=16  # p5 has stride 16
    )

    p7_boxes, p7_scores, p7_class_ids = decode_dfl_predictions(
        p7_pred["bbox"], 
        p7_pred["cls"], 
        stride=32  # p7 has stride 32
    )

    # Combine all predictions
    all_boxes = p3_boxes + p5_boxes + p7_boxes
    all_scores = p3_scores + p5_scores + p7_scores
    all_class_ids = p3_class_ids + p5_class_ids + p7_class_ids

    return all_boxes, all_scores, all_class_ids

# Decode predictions from p3, p5, and p7
boxes, scores, class_ids = decode_all_predictions(p3_pred, p5_pred, p7_pred)

# Apply NMS
if boxes:
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    
    final_boxes = [boxes[i] for i in keep]
    final_scores = [scores[i] for i in keep]
    final_class_ids = [class_ids[i] for i in keep]
else:
    final_boxes, final_scores, final_class_ids = [], [], []

# Visualization
img_np = img.squeeze().permute(1, 2, 0).numpy()
fig, ax = plt.subplots(1)
ax.imshow(img_np)

for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
    x_min, y_min, x_max, y_max = box
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x_min, y_min - 5, f"Cls {class_id}: {score:.2f}", color='white',
            fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

plt.show()
