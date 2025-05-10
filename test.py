import torch
from models import YOLO
from dataset import YOLOv8Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from utils import plot_boxes


if __name__ == "__main__":
    MODEL_DIR = r"weights/initial_yolo_last.pth"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    train_set = YOLOv8Dataset("data/ship-detection_6", "val", grid_size=7, transform=transform, normalize=False, augment=False)
    classlist = train_set.classes
    model = YOLO(len(classlist))
    model.eval()
    checkpoint = torch.load(MODEL_DIR, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        for data, targets, original_data in train_set:
            data = data.unsqueeze(0)
            output = model(data)
            output = tuple(t.squeeze(0) for t in output)
            plot_boxes(original_data, *output, class_names=classlist, conf_threshold=0.5, max_overlap=0.3)