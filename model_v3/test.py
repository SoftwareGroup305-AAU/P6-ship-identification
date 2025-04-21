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
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Paths
model_file = r"yolo_custom_best.pth"
test_img = r"yolo/data/test/images/images.jpeg"

# Load and preprocess the image
img = read_image(test_img)
img = train_transforms(img).unsqueeze(0)

# Initialize model
model = YOLO(num_classes=11)
model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
model.eval()

# Forward pass
with torch.no_grad():
    predictions = model(img)

# Decode function with correct stride and clamping/sanity filtering
def decode_dfl_predictions(pred_bbox, pred_cls, reg_max=16, conf_thresh=0.3, stride=8, img_size=640):
    B, _, H, W = pred_bbox.shape
    device = pred_bbox.device

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

                x_min = max(0, min(x_center - l * stride, img_size))
                y_min = max(0, min(y_center - t * stride, img_size))
                x_max = max(0, min(x_center + r * stride, img_size))
                y_max = max(0, min(y_center + b_ * stride, img_size))

                w = x_max - x_min
                h = y_max - y_min

                class_logits = pred_cls[b, :, i, j]
                class_scores = torch.softmax(class_logits, dim=0)
                score, class_id = torch.max(class_scores, dim=0)

                if score.item() > conf_thresh and w > 1 and h > 1:
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append(score.item())
                    class_ids.append(class_id.item())

    return boxes, scores, class_ids

# Decode and combine predictions from all scales with correct stride per scale
all_boxes, all_scores, all_class_ids = [], [], []
scale_strides = {"p3": 8, "p5": 16, "p7": 32}
for scale in ["p3", "p5", "p7"]:
    b, s, c = decode_dfl_predictions(
        predictions[scale]["bbox"], predictions[scale]["cls"], stride=scale_strides[scale]
    )
    all_boxes.extend(b)
    all_scores.extend(s)
    all_class_ids.extend(c)

# Apply NMS across all boxes
if all_boxes:
    boxes_tensor = torch.tensor(all_boxes)
    scores_tensor = torch.tensor(all_scores)
    keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

    final_boxes = [all_boxes[i] for i in keep]
    final_scores = [all_scores[i] for i in keep]
    final_class_ids = [all_class_ids[i] for i in keep]
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
