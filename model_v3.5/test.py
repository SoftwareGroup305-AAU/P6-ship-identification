from torchvision.io import read_image
from torchvision import transforms
from core import YOLO
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms

# Define transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Paths
model_file = r"model_v3.5/server/modelsBackup/yolo_custom_last.pth"
test_img    = r"sail_boat.jpg"

# Load and preprocess the image
img = read_image(test_img)
orig_size = img.shape[1], img.shape[2]
img = preprocess(img).unsqueeze(0)

# Initialize model -- must match the architecture used during training
model = YOLO(num_classes=11)
# Note: core.YOLO only takes num_classes (and optional reg_max) in this version.

# Load weights (allow missing/unexpected keys if trimming neck)
state = torch.load(model_file, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

# Forward pass
with torch.no_grad():
    predictions = model(img)

# Decode function (no change)
def decode_dfl_predictions(pred_bbox, pred_cls, reg_max=16, conf_thresh=0.25, stride=8, img_size=640):
    B, _, H, W = pred_bbox.shape
    device = pred_bbox.device

    # DFL decode
    pred = pred_bbox.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2)
    pred = F.softmax(pred, dim=-1)
    pred = pred @ torch.arange(reg_max, device=device, dtype=torch.float)

    boxes, scores, classes = [], [], []
    for b in range(B):
        for i in range(H):
            for j in range(W):
                l, t, r, bot = pred[b, i, j]
                x_c = (j + 0.5) * stride
                y_c = (i + 0.5) * stride
                x1, y1 = x_c - l*stride, y_c - t*stride
                x2, y2 = x_c + r*stride, y_c + bot*stride
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_size, x2), min(img_size, y2)
                w, h = x2-x1, y2-y1
                if w>1 and h>1:
                    logits = pred_cls[b, :, i, j]
                    prob   = torch.softmax(logits, dim=0)
                    score, cls_id = prob.max(0)
                    if score>conf_thresh:
                        boxes.append([x1,y1,x2,y2])
                        scores.append(score.item())
                        classes.append(cls_id.item())
    return boxes, scores, classes

# Combine predictions
all_boxes, all_scores, all_classes = [], [], []
strides = {'p3':8, 'p5':16, 'p7':32}
for scale in ['p3','p5','p7']:
    bbox = predictions[scale]['bbox']       # [B,4*reg_max,H,W]
    cls  = predictions[scale]['cls']        # [B,H,W,nc]
    cls  = cls.permute(0,3,1,2)             # -> [B,nc,H,W]
    b, s, c = decode_dfl_predictions(bbox, cls, stride=strides[scale])
    all_boxes.extend(b)
    all_scores.extend(s)
    all_classes.extend(c)

# NMS
if all_boxes:
    boxes_t = torch.tensor(all_boxes)
    scores_t= torch.tensor(all_scores)
    keep = nms(boxes_t, scores_t, iou_threshold=0.5)
    final = [(all_boxes[i], all_scores[i], all_classes[i]) for i in keep]
else:
    final = []

# Visualization
img_vis = preprocess(read_image(test_img)).permute(1,2,0).numpy()
fig, ax = plt.subplots()
ax.imshow(img_vis)
for (x1,y1,x2,y2), sc, cl in final:
    rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-4, f"{cl}:{sc:.2f}", color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.6))
plt.axis('off')
plt.show()
