import torch
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from core import YOLOv1

# names = ['Boat', 'Cargo-Ship', 'Carrier-Ship', 'Container-Ship', 'Cruise-Ship', 'Fish-Boat', 'Sail-Boat', 'Submarine', 'Tanker-Ship', 'Tugboat', 'War-Ship']

names = ['container', 'cruise', 'fish-b', 'sail boat', 'submarine', 'warship']

def visualize_predictions(model_path, image_path, conf_threshold=0.1, num_bboxes=2, num_classes=11, debug=True):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = YOLOv1(number_of_bboxes=2, number_of_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    
    # Get original dimensions
    original_width, original_height = pil_img.size
    
    # Transform for model
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(img_tensor)  # (1, 7, 7, B*5 + C)
    
    preds = preds[0].cpu()  # (7, 7, B*5 + C)
    S = 7
    boxes = []
    classes = []
    scores = []

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(np.array(pil_img))


    for i in range(S):
        for j in range(S):
            cell = preds[i, j]
            class_scores = cell[num_bboxes*5:]
            class_id = torch.argmax(class_scores).item()
            class_conf = torch.softmax(class_scores, dim=0)[class_id].item()
            
            for b in range(num_bboxes):
                offset = b * 5
                obj_score = cell[offset + 4].item()
                score = obj_score * class_conf
                if score < conf_threshold:
                    continue

                cx = (cell[offset + 0].item() + j) / S
                cy = (cell[offset + 1].item() + i) / S
                w = cell[offset + 2].item()
                h = cell[offset + 3].item()

                x1 = max(0, (cx - w / 2) * original_width)
                y1 = max(0, (cy - h / 2) * original_height)
                x2 = min(original_width, (cx + w / 2) * original_width)
                y2 = min(original_height, (cy + h / 2) * original_height)

                boxes.append([x1, y1, x2, y2])
                classes.append(class_id)
                scores.append(score)

                width, height = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(x1, y1 - 5, f"{names[class_id]}: {score:.2f}",
                         color='white', fontsize=10,
                         bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title(f"Detected {len(boxes)} objects with confidence > {conf_threshold}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Placeholders for model and image paths
    MODEL_PATH = "simple_yolo_custom_last.pth"
    IMAGE_PATH = "bottom_trawler.jpg"
    
    # Lowered confidence threshold to see more detections
    visualize_predictions(
        model_path=MODEL_PATH, 
        image_path=IMAGE_PATH, 
        conf_threshold=0.05,  # Reduced from 0.4 to 0.1
        num_bboxes=2,
        num_classes=6,
        debug=True  # Enable debug information
    )
