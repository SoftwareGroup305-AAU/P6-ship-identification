import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import yaml

class YOLOv8Dataset(Dataset):
    def __init__(self, data_dir, image_set, grid_size, transform, normalize = False, augment = False):
        
        with open(os.path.join(data_dir, "data.yaml"), "r") as file:
            config = yaml.safe_load(file)

        set_path = config[image_set].lstrip(os.sep)
        self.image_dir = os.path.join(data_dir, set_path)
        self.label_dir = os.path.join(data_dir, set_path.replace("images", "labels"))        
        self.transform = transform
        self.normalize = normalize
        self.augment = augment
        self.classes = config["names"]
        self.S = grid_size
        self.C = config["nc"]
        self.image_files = [files for files in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = read_image(image_path)
        original_image = image

        if self.transform:
            image = self.transform(image)

        label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    values = [float(value) for value in line.split()]
                    labels.append(values)
        objectness_target = torch.zeros((self.S, self.S))
        class_target = torch.zeros((self.S, self.S, self.C))
        localization_target = torch.zeros((self.S, self.S, 4))
        for label in labels:
            c, x, y, w, h = label
            c = int(c)
            
            grid_x = int(x * self.S)
            grid_y = int(y * self.S)

            objectness_target[grid_y, grid_x] = 1
            class_target[grid_y, grid_x, c] = 1
            localization_target[grid_y, grid_x] = torch.tensor([x,y,w,h])
        return image, (class_target, objectness_target, localization_target), original_image
    
    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        class_targets, objectness_targets, localization_targets = zip(*targets)

        images = torch.stack(images)
        class_targets = torch.stack(class_targets)
        objectness_targets = torch.stack(objectness_targets)
        localization_targets = torch.stack(localization_targets)

        return images, (class_targets, objectness_targets, localization_targets)

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from utils import plot_boxes_from_yolo_target

if __name__ == "__main__":

    transform = T.Compose([
        T.Resize((448, 448))
    ])

    train_set = YOLOv8Dataset("data/ship-detection", "train", grid_size=7, transform=transform, normalize=False, augment=False)
    classlist = train_set.classes
    for data, targets in train_set:
        plot_boxes_from_yolo_target(data, *targets, conf_threshold=0.5, class_names=classlist)

            