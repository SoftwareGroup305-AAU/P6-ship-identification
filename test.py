import torch
from models import YOLO
from dataset import YOLOv8Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from utils import plot_boxes


if __name__ == "__main__":
    MODEL_DIR = r"weights/final.pth"

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ConvertImageDtype(torch.float)
    ])
    train_set = YOLOv8Dataset("data/ship-detection-6", "train", grid_size=7, transform=transform, normalize=False, augment=False)
    classlist = train_set.classes
    model = YOLO(len(classlist))
    model.eval()
    checkpoint = torch.load(MODEL_DIR, map_location="cpu")
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        for data, targets in train_set:
            data = data.unsqueeze(0)
            output = model(data)
            data = data.squeeze(0)
            output = tuple(t.squeeze(0) for t in output)
            plot_boxes(data, *output, class_names=classlist, conf_threshold=0.5, max_overlap=0.2)