import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [files for files in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = read_image(image_path) # could normalize, dont know if necessary after transform
        
        label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".txt")
        target = []
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    values = [float(value) for value in line.split()]
                    target.append(values)
        if target:
            target = torch.tensor(target, dtype=torch.float32)
        else:
            target = torch.zeros((0, 5))
        
        if self.transform:
            image = self.transform(image)
        
        return image, target