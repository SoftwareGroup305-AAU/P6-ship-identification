import torch
from models import YOLO
from dataset import YOLOv8Dataset
from PIL import Image
import torchvision.transforms as T
from utils import plot_boxes


if __name__ == "__main__":
    MODEL_DIR = r"weights/final.pth"
    IMG_DIR = "test_images/sail-boats.jpg"

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((448, 448)),
        T.ConvertImageDtype(torch.float)
    ])

    image = Image.open(IMG_DIR).convert("RGB")
    image = transform(image).unsqueeze(0)
    classlist = ["container", "cruise", "fish-b", "sail boat", "submarine", "warship"]
    model = YOLO(len(classlist))
    model.eval()
    checkpoint = torch.load(MODEL_DIR, map_location="cpu")
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        output = model(image)
        output = tuple(t.squeeze(0) for t in output)
        image = image.squeeze(0)
        plot_boxes(image, *output, class_names=classlist, conf_threshold=0.5, max_overlap=0.2)
            