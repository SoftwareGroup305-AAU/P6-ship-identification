import torch
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from core import YOLO
from torchvision.ops import box_iou

# names = ['Boat', 'Cargo-Ship', 'Carrier-Ship', 'Container-Ship', 'Cruise-Ship', 'Fish-Boat', 'Sail-Boat', 'Submarine', 'Tanker-Ship', 'Tugboat', 'War-Ship']

names = ['container', 'cruise', 'fish-b', 'sail boat', 'submarine', 'warship']

def visualize_predictions(model_path, image_path, conf_threshold=0.1, num_classes=6, max_overlap=0.5, debug=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Image transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    
    # Load image
    img = read_image(image_path)
    pil_img = Image.open(image_path)
    original_width, original_height = pil_img.size
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        class_pred, obj_pred, loc_pred = model(img_tensor)
    
    if debug:
        print(f"Max objectness score: {obj_pred.max().item():.4f}")
        print(f"Min objectness score: {obj_pred.min().item():.4f}")
        print(f"Mean objectness score: {obj_pred.mean().item():.4f}")
        plt.figure(figsize=(10, 8))
        plt.title("Objectness Confidence Heatmap")
        plt.imshow(obj_pred[0].cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Confidence')
        plt.show()
    
    grid_size = obj_pred.shape[1]
    raw_boxes = []

    # Collect all raw boxes above confidence threshold
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            confidence = obj_pred[0, grid_y, grid_x].item()
            class_scores = class_pred[0, grid_y, grid_x]
            class_id = torch.argmax(class_scores).item()

            if num_classes <= 2:
                class_conf = torch.sigmoid(class_scores[class_id]).item()
            else:
                class_conf = torch.softmax(class_scores, dim=0)[class_id].item()

            score = confidence * class_conf

            if score > conf_threshold:
                box = loc_pred[0, grid_y, grid_x]
                cx, cy, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()

                cx = (cx + grid_x) / grid_size
                cy = (cy + grid_y) / grid_size

                x1 = max(0, (cx - w / 2) * original_width)
                y1 = max(0, (cy - h / 2) * original_height)
                x2 = min(original_width, (cx + w / 2) * original_width)
                y2 = min(original_height, (cy + h / 2) * original_height)

                raw_boxes.append((class_id, score, [x1, y1, x2, y2]))

    # Apply per-class non-maximum suppression
    raw_boxes.sort(key=lambda x: x[1], reverse=True)
    kept_boxes = []
    used = [False] * len(raw_boxes)

    for i in range(len(raw_boxes)):
        if used[i]:
            continue
        class_i, score_i, box_i = raw_boxes[i]
        kept_boxes.append(raw_boxes[i])
        for j in range(i + 1, len(raw_boxes)):
            if used[j]:
                continue
            class_j, score_j, box_j = raw_boxes[j]
            if class_i == class_j:
                iou = box_iou(torch.tensor([box_i]), torch.tensor([box_j]))[0, 0].item()
                if iou > max_overlap:
                    used[j] = True

    # Draw boxes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(np.array(pil_img))
    
    boxes = []
    classes = []
    scores = []

    for class_id, score, (x1, y1, x2, y2) in kept_boxes:
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        label = f"Class {class_id}: {score:.2f}"
        plt.text(x1, y1 - 5, label, color='white', fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.5))

        boxes.append([x1, y1, x2, y2])
        classes.append(class_id)
        scores.append(score)

    plt.title(f"Detected {len(boxes)} objects with NMS + confidence > {conf_threshold}")
    plt.axis('off')
    plt.show()
    
    return boxes, classes, scores

if __name__ == "__main__":
    # Placeholders for model and image paths
    MODEL_PATH = "model/weights/initial_yolo_last.pth"
    IMAGE_PATH = "model/warship2.jpg"
    
    # Lowered confidence threshold to see more detections
    visualize_predictions(
        model_path=MODEL_PATH, 
        image_path=IMAGE_PATH, 
        conf_threshold=0.5  ,
        num_classes=6,
        max_overlap=0.1,
        debug=True 
    )
