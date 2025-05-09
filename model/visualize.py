import torch
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from core import YOLO

# names = ['Boat', 'Cargo-Ship', 'Carrier-Ship', 'Container-Ship', 'Cruise-Ship', 'Fish-Boat', 'Sail-Boat', 'Submarine', 'Tanker-Ship', 'Tugboat', 'War-Ship']

names = ['container', 'cruise', 'fish-b', 'sail boat', 'submarine', 'warship']

def visualize_predictions(model_path, image_path, conf_threshold=0.1, num_classes=6, debug=True):
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
    
    # Get original dimensions
    original_width, original_height = pil_img.size
    
    # Transform for model
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        class_pred, obj_pred, loc_pred = model(img_tensor)
    
    # Print debugging information
    if debug:
        print(f"Max objectness score: {obj_pred.max().item():.4f}")
        print(f"Min objectness score: {obj_pred.min().item():.4f}")
        print(f"Mean objectness score: {obj_pred.mean().item():.4f}")
        
        # Plot objectness heatmap
        plt.figure(figsize=(10, 8))
        plt.title("Objectness Confidence Heatmap")
        plt.imshow(obj_pred[0].cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Confidence')
        plt.show()
    
    # Extract bounding boxes
    grid_size = obj_pred.shape[1]
    boxes = []
    classes = []
    scores = []
    
    # Create figure and axis for plotting
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(np.array(pil_img))
    
    # Process predictions and extract bounding boxes
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            confidence = obj_pred[0, grid_y, grid_x].item()
            
            if confidence > conf_threshold:
                # Get class prediction
                class_scores = class_pred[0, grid_y, grid_x]
                class_id = torch.argmax(class_scores).item()
                
                # Calculate final confidence score
                if num_classes <= 2:
                    class_conf = torch.sigmoid(class_scores[class_id]).item()
                else:
                    class_conf = torch.softmax(class_scores, dim=0)[class_id].item()
                
                score = confidence * class_conf
                
                # Get box coordinates (center_x, center_y, width, height)
                box = loc_pred[0, grid_y, grid_x]
                cx, cy, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                
                # Normalize coordinates to grid
                cx = (cx + grid_x) / grid_size  # Keep this
                cy = (cy + grid_y) / grid_size  # Keep this
                w = w  # Remove division by grid_size
                h = h # Remove division by grid_siz
                
                # Convert to original image coordinates
                x1 = max(0, (cx - w/2) * original_width)
                y1 = max(0, (cy - h/2) * original_height)
                x2 = min(original_width, (cx + w/2) * original_width)
                y2 = min(original_height, (cy + h/2) * original_height)
                
                # Store box and class info
                boxes.append([x1, y1, x2, y2])
                classes.append(class_id)
                scores.append(score)
                
                # Draw rectangle on image
                width, height = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, 
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                plt.text(x1, y1-5, f"Class {names[class_id]}: {score:.2f}", 
                        color='white', fontsize=10, 
                        bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title(f"Detected {len(boxes)} objects with confidence > {conf_threshold}")
    plt.axis('off')
    plt.show()
    
    return boxes, classes, scores

if __name__ == "__main__":
    # Placeholders for model and image paths
    MODEL_PATH = "initial_yolo_last.pth"
    IMAGE_PATH = "sail_boat.jpg"
    
    # Lowered confidence threshold to see more detections
    visualize_predictions(
        model_path=MODEL_PATH, 
        image_path=IMAGE_PATH, 
        conf_threshold=0.9,
        num_classes=6,
        debug=True 
    )
