import torch
import torch.utils.data as data
import torchvision.datasets.voc as voc
import os
import json
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

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
    """Plots bounding boxes on the given image using matplotlib."""
    C = len(classes)
    grid_size_x = data.size(dim=2) / 7
    grid_size_y = data.size(dim=1) / 7
    m = labels.size(dim=0)
    n = labels.size(dim=1)

    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((labels.size(dim=2) - C) // 5):
                bbox_start = 5 * k + C
                bbox_end = 5 * (k + 1) + C
                bbox = labels[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(labels[i, j, :C]).item()
                confidence = labels[i, j, class_index].item() * bbox[4].item()  # pr(c) * IOU
                if confidence > min_confidence:
                    width = bbox[2] * 448
                    height = bbox[3] * 448
                    tl = (
                        bbox[0] * 448 + j * grid_size_x - width / 2,
                        bbox[1] * 448 + i * grid_size_y - height / 2
                    )
                    bboxes.append([tl, width, height, confidence, class_index])

    # Sort by highest to lowest confidence
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)

    # Calculate IOUs between each pair of boxes
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Non-maximum suppression
    discarded = set()
    fig, ax = plt.subplots(1, figsize=(12, 12))  # Create a single plot
    ax.imshow(data.permute(1, 2, 0).cpu().numpy())  # Convert from tensor to numpy for imshow
    ax.axis('off')

    for i in range(num_boxes):
        if i not in discarded:
            tl, width, height, confidence, class_index = bboxes[i]

            # Decrease confidence of other conflicting bboxes
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Annotate image
            rect = patches.Rectangle(
                tl, width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Draw text label
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            ax.text(tl[0], tl[1] - 10, text, color='orange', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    if file is None:
        plt.show()
    else:
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        plt.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the plot after saving or showing


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

    train_set = YOLOPascalVoc("data", '2012', "train", grid_size=7, num_predictors=2, transform=transform, normalize=True, augment=True)
    classes = train_set.class_dict
    classlist = load_class_array(classes)

    for data, label, _  in train_set:
        plot_ground_truths(data, label, classlist, max_overlap=float('inf'))
        
