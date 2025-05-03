import torch
import torch.utils.data as data
import torchvision.datasets.voc as voc
import os
import json
import torchvision.transforms as T


class YOLOPascalVoc(data.Dataset):
    def __init__(self, data_path, year, image_set, grid_size, num_predictors, num_classes, transform = None, augment = None,):
        super().__init__()
        assert image_set in {"train", "val", "test"}
        assert year in {"2007", "2012"}
        self.augment = augment
        self.S = grid_size
        self.B = num_predictors
        self.C = num_classes
        self.depth = self.B*5+self.C
        self.dataset = voc.VOCDetection(root=data_path,
                                         year=year, 
                                         image_set=image_set, 
                                         download=False, 
                                         transform=transform)
        class_path = os.path.join(data_path, "classes.json")
        if not os.path.exists(class_path):
            self.class_dict = self._build_classes()
            self._save_classes(self.class_dict, class_path)
        else:
            self.class_dict = self._load_classes(class_path)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        image_height, image_width = data.shape[-2:]
        if self.augment:
            self.augment(data)

        grid_size_x = image_width / self.S
        grid_size_y =  image_height / self.S

        target = torch.zeros((self.S, self.S, self.depth))

        raw_bboxes_pairs = self._get_rescaled_bounding_boxes(label, image_width, image_height)

        for bbox_pair in raw_bboxes_pairs:
            name, coords = bbox_pair
            class_index = self.class_dict[name]
            x_min, x_max, y_min, y_max = coords

            center_x = (x_max + x_min) / 2
            center_y = (y_max + y_min) / 2
            grid_x = int(center_x // grid_size_x)
            grid_y = int(center_y // grid_size_y)
            
            if 0 <= grid_x < self.S and 0 <= grid_y < self.S:
                pass
    
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

if __name__ == '__main__':
    ### TESTING ###
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((448, 448))
        ]
    )
    augment = T.Compose([
            T.RandomAffine(
                degrees=0,  # No rotation
                translate=(0.1, 0.1),  # ±10% shift
                scale=(0.9, 1.2),  # Equivalent to 1 ± 0.2
                shear=0.0
            ),
            T.ColorJitter(
                hue=0.1,  # ±0.1
                saturation=(0.9, 1.1)  # Approx. 0.9 to 1.1
            ),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    dataset = YOLOPascalVoc("data", "2012", "train", transform=transform, augment=augment, grid_size=7, num_classes=20, num_predictors=2 )
    hello = dataset.__getitem__(50)