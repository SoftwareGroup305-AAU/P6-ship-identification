import torch
import torch.utils.data as data
import torchvision.datasets.voc as voc
import os
import json
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw


class YOLOPascalVoc(data.Dataset):
    def __init__(self, data_path, year, image_set, grid_size, num_predictors, transform = None, augment = True, normalize=True):
        super().__init__()
        assert image_set in {"train", "val", "test"}
        assert year in {"2007", "2012"}
        self.augment = augment
        self.normalize = normalize
        self.S = grid_size
        self.B = num_predictors
        self.dataset = voc.VOCDetection(root=data_path,
                                         year=year, 
                                         image_set=image_set, 
                                         download=True, 
                                         transform=transform)
        class_path = os.path.join(data_path, "classes.json")
        if not os.path.exists(class_path):
            self.class_dict = self._build_classes()
            self._save_classes(self.class_dict, class_path)
        else:
            self.class_dict = self._load_classes(class_path)

        self.C = len(self.class_dict)
        self.depth = self.B*5+self.C

    def __getitem__(self, index):
        data, label = self.dataset[index]
        original_data = data
        image_height, image_width = data.shape[-2:]
        x_shift = int((0.2 * random.random() - 0.1) * image_width)
        y_shift = int((0.2 * random.random() - 0.1) * image_height)
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        grid_size_x = image_width / self.S
        grid_size_y =  image_height / self.S

        target = torch.zeros((self.S, self.S, self.depth))

        raw_bboxes_pairs = self._get_rescaled_bounding_boxes(label, image_width, image_height)

        cell_box_counts = {}   # Tracks how many boxes have been assigned per cell
        cell_class_flags = {} # Tracks if a class has been assigned to a cell

        for name, (x_min, x_max, y_min, y_max) in raw_bboxes_pairs:
            if name not in self.class_dict:
                continue  # skip unknown classes

            class_idx = self.class_dict[name]

            # Augment labels if needed
            if self.augment:
                half_w, half_h = image_width / 2, image_height / 2
                x_min = self._scale_bbox_coord(x_min, half_w, scale) + x_shift
                x_max = self._scale_bbox_coord(x_max, half_w, scale) + x_shift
                y_min = self._scale_bbox_coord(y_min, half_h, scale) + y_shift
                y_max = self._scale_bbox_coord(y_max, half_h, scale) + y_shift

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            cell_x = int(center_x // grid_size_x)
            cell_y = int(center_y // grid_size_y)
            cell = (cell_y, cell_x)

            if not (0 <= cell_x < self.S and 0 <= cell_y < self.S):
                continue  # skip boxes falling outside grid

            # Set class one-hot only once per cell
            if cell not in cell_class_flags:
                one_hot = torch.zeros(self.C)
                one_hot[class_idx] = 1.0
                target[cell_y, cell_x, :self.C] = one_hot
                cell_class_flags[cell] = name

            # Assign bounding box
            box_count = cell_box_counts.get(cell, 0)
            if box_count < self.B:
                rel_x = (center_x - cell_x * grid_size_x) / image_width
                rel_y = (center_y - cell_y * grid_size_y) / image_height
                rel_w = (x_max - x_min) / image_width
                rel_h = (y_max - y_min) / image_height

                start = self.C + box_count * 5
                target[cell_y, cell_x, start:start+5] = torch.tensor([rel_x, rel_y, rel_w, rel_h, 1.0])
                cell_box_counts[cell] = box_count + 1
        return data, target, original_data
                

    def _scale_bbox_coord(self, coord, center, scale):
        return ((coord - center) * scale) + center 
    
    def _load_classes(self, class_path):
        if os.path.exists(class_path):
            with open(class_path, "r") as file:
                return json.load(file)
        return {} # no classes.json file
        
    def _save_classes(self, class_dict, class_path):
        with open(class_path, "w") as file:
            json.dump(class_dict, file, indent=2)

    def _build_classes(self):
        assert self.dataset is not None
        class_dict = {}
        index = 0
        for i, (data, label) in enumerate(self.dataset):
            objects = label["annotation"]["object"]
            if isinstance(objects, dict):
                objects = [objects]
            for object in objects:
                name = object["name"]
                if name not in class_dict:
                    class_dict[name] = index
                    index += 1
        return class_dict
    
    def _get_rescaled_bounding_boxes(self, label, image_width, image_height):
        size = label['annotation']['size']
        width, height = int(size['width']), int(size['height'])
        x_scale = image_width / width
        y_scale = image_height / height
        boxes = []
        objects = label['annotation']['object']
        for obj in objects:
            box = obj['bndbox']
            coords = (
                int(int(box['xmin']) * x_scale),
                int(int(box['xmax']) * x_scale),
                int(int(box['ymin']) * y_scale),
                int(int(box['ymax']) * y_scale)
            )
            name = obj['name']
            boxes.append((name, coords))
        return boxes
    
    def __len__(self):
        return len(self.dataset)
    

def get_overlap(a, b):
    """Returns proportion overlap between two boxes in the form (tl, width, height, confidence, class)."""

    a_tl, a_width, a_height, _, _ = a
    b_tl, b_width, b_height, _, _ = b

    i_tl = (
        max(a_tl[0], b_tl[0]),
        max(a_tl[1], b_tl[1])
    )
    i_br = (
        min(a_tl[0] + a_width, b_tl[0] + b_width),
        min(a_tl[1] + a_height, b_tl[1] + b_height),
    )

    intersection = max(0, i_br[0] - i_tl[0]) \
                   * max(0, i_br[1] - i_tl[1])

    a_area = a_width * a_height
    b_area = b_width * b_height

    a_intersection = b_intersection = intersection
    if a_area == 0:
        a_intersection = 0
        a_area = 1E-6
    if b_area == 0:
        b_intersection = 0
        b_area = 1E-6

    return torch.max(
        a_intersection / a_area,
        b_intersection / b_area
    ).item()

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_ground_truths(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None):
    """Plots bounding boxes on the given image using matplotlib.
    
    Args:
        data: Image tensor (C, H, W) - augmented if dataset.augment=True
        labels: Target tensor (S, S, B*5+C) with bounding boxes
        classes: List of class names
        color: Color for bounding boxes
        min_confidence: Minimum confidence threshold to display boxes
        max_overlap: Maximum allowed overlap between boxes
        file: Optional file path to save the plot
    """
    # Denormalize the image if it's normalized
    if torch.min(data) < 0:  # Simple check for normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(data.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(data.device)
        data = data * std + mean  # Reverse normalization
        data = torch.clamp(data, 0, 1)  # Ensure valid pixel range
    
    # Convert tensor to numpy and permute dimensions for matplotlib
    img = data.permute(1, 2, 0).cpu().numpy()
    
    # Calculate grid cell sizes
    img_height, img_width = data.shape[1], data.shape[2]
    grid_size_x = img_width / labels.shape[1]  # S
    grid_size_y = img_height / labels.shape[0]  # S
    
    C = len(classes)
    B = (labels.shape[2] - C) // 5  # Number of predictors per cell
    
    # Collect all bounding boxes
    bboxes = []
    for i in range(labels.shape[0]):  # rows (S)
        for j in range(labels.shape[1]):  # cols (S)
            for k in range(B):  # predictors
                bbox_start = C + k * 5
                bbox = labels[i, j, bbox_start:bbox_start+5]
                confidence = bbox[4].item()
                
                if confidence > min_confidence:
                    # Get class with highest probability
                    class_probs = labels[i, j, :C]
                    class_idx = torch.argmax(class_probs).item()
                    
                    # Convert relative coordinates to absolute
                    rel_x, rel_y, rel_w, rel_h = bbox[:4]
                    width = rel_w * img_width
                    height = rel_h * img_height
                    center_x = rel_x * img_width + j * grid_size_x
                    center_y = rel_y * img_height + i * grid_size_y
                    
                    # Calculate top-left corner
                    x = center_x - width / 2
                    y = center_y - height / 2
                    
                    bboxes.append([
                        (x, y),        # top-left coordinates
                        width,         # box width
                        height,        # box height
                        confidence,    # confidence score
                        class_idx      # class index
                    ])

    # Non-maximum suppression
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)  # Sort by confidence
    keep = []
    
    while len(bboxes) > 0:
        current = bboxes.pop(0)
        keep.append(current)
        
        # Remove overlapping boxes of the same class
        bboxes = [
            box for box in bboxes 
            if box[4] != current[4] or  # Different class
            get_overlap(current, box) <= max_overlap  # Low overlap
        ]

    # Create plot
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)
    ax.axis('off')

    # Draw kept boxes
    for (x, y), width, height, confidence, class_idx in keep:
        # Draw bounding box
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw label
        label = f"{classes[class_idx]} {confidence:.1%}"
        ax.text(
            x, y - 10, label,
            color=color,
            fontsize=10,
            bbox=dict(
                facecolor='black',
                alpha=0.5,
                edgecolor='none',
                boxstyle='round,pad=0.2'
            )
        )

    # Save or display
    if file is not None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close(fig)

def load_class_array(classes):
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result

if __name__ == '__main__':
    # Display data
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((448, 448))
    ])

    train_set = YOLOPascalVoc("data", '2007', "train", grid_size=7, num_predictors=2, transform=transform, normalize=True, augment=True)
    classes = train_set.class_dict
    classlist = load_class_array(classes)

    for data, label, _  in train_set:
         utils.plot_boxes(data, label, classes, max_overlap=float('inf'))
        
