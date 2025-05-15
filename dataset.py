import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import utils
import yaml
import torchvision.transforms.functional as TF

class YOLOv8Dataset(Dataset):
    def __init__(self, data_dir, image_set, grid_size, transform: utils.YoloAugment | None = None, raw_labels=False):
        
        with open(os.path.join(data_dir, "data.yaml"), "r") as file:
            config = yaml.safe_load(file)

        set_path = config[image_set].lstrip(os.sep)
        self.image_dir = os.path.join(data_dir, set_path)
        self.label_dir = os.path.join(data_dir, set_path.replace("images", "labels"))        
        self.transform = transform
        self.classes = config["names"]
        self.S = grid_size
        self.C = config["nc"]
        self.image_files = [files for files in os.listdir(self.image_dir)]
        self.raw_labels = raw_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = read_image(image_path).float() / 255.0
        label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    values = [float(value) for value in line.split()]
                    labels.append(values)
        if self.transform:
            image, labels = self.transform(image, labels)
        if self.raw_labels:
            return image, labels
        objectness_target = torch.zeros((self.S, self.S))
        class_target = torch.zeros((self.S, self.S, self.C))
        localization_target = torch.zeros((self.S, self.S, 4))
        for label in labels:
            c, x, y, w, h = label
            c = int(c)
            x = min(max(x, 0), 1 - 1e-6) # clamp if out of range
            y = min(max(y, 0), 1 - 1e-6)
            grid_x = int(x * self.S)
            grid_y = int(y * self.S)

            objectness_target[grid_y, grid_x] = 1
            class_target[grid_y, grid_x, c] = 1
            localization_target[grid_y, grid_x] = torch.tensor([x,y,w,h])
        return image, (class_target, objectness_target, localization_target)

from utils import plot_boxes

if __name__ == "__main__":
    augment = utils.YoloAugment(resize=(448,448), rotate_prob=0.8, max_rotate_angle=50)
    train_set = YOLOv8Dataset("data/ship-detection-6-neg", "train", grid_size=7, transform=augment)
    classlist = train_set.classes
    for data, targets in train_set:
        plot_boxes(data, *targets, conf_threshold=0.5, class_names=classlist)

            