import torch
import os
import utils
from tqdm import tqdm
from dataset import YOLOPascalVoc, plot_ground_truths
from models import *
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from collections import OrderedDict
import torchvision.transforms.functional as TF


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

MODEL_DIR = 'weights/epoch_100.pth'
IMG_DIR = r"images/ship.jpg"
CLASS_FILE = "data/classes.json"

def plot_test_images():

    classes = utils.load_class_array(CLASS_FILE)
    C = len(classes)
    model = YOLOv1ResNet()
    model.eval()
    state_dict = torch.load(MODEL_DIR, map_location="cpu")
    model.load_state_dict(state_dict)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((448, 448))
    ])

    dataset = YOLOPascalVoc('data', "2007", "test", grid_size=7, num_predictors=2, transform=transform, normalize=True, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    img = Image.open(IMG_DIR).convert("RGB")

    

    img = transform(img)
    img = img.unsqueeze(0)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    with torch.no_grad():
        prediction = model(img)

        img = img.squeeze(0)
        prediction = prediction.squeeze(0)

        plot_ground_truths(img, prediction, classes, max_overlap=0.5, min_confidence=0.1)

        for image, labels, original in tqdm(loader):
            predictions = model.forward(image)
            predictions = predictions.squeeze(0)
            image = image.squeeze(0)
            plot_ground_truths(
                    image,
                    predictions,
                    classes,
                    max_overlap=0.5, 
                    min_confidence=0.4
                )


if __name__ == '__main__':
    plot_test_images()
