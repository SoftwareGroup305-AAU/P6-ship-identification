from torchvision import transforms
import torchvision.transforms.functional as F
import random
import torch
import math

class CustomYOLOTransform:
    def __init__(self, size=640):
        self.size = size
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.blur = transforms.GaussianBlur(kernel_size=3)

    def __call__(self, image, target):
        image = F.to_pil_image(image)
        image = F.resize(image, (self.size, self.size))

        if random.random() < 0.85:
            image = self.color_jitter(image)
            image = F.rotate(image, angle=random.uniform(-15, 15))
            image = self.blur(image)

        if random.random() < 0.5:
            image = F.hflip(image)
            target[:, 1] = 1 - target[:, 1]

        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
        target = self.rotate_bboxes(target, angle)

        scale = random.uniform(0.7, 1.0)
        crop_size = int(self.size * scale)
        top = random.randint(0, self.size - crop_size)
        left = random.randint(0, self.size - crop_size)

        image = F.resized_crop(image, top, left, crop_size, crop_size, (self.size, self.size))
        target = self.crop_resize_bboxes(target, left, top, crop_size, self.size)

        image = F.to_tensor(image)

        return image, target

    def rotate_bboxes(self, target, angle):
        angle_rad = -angle * (math.pi / 180)

        cx = target[:, 1] * self.size
        cy = target[:, 2] * self.size
        w = target[:, 3] * self.size
        h = target[:, 4] * self.size

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        corners = torch.stack([
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y1], dim=1),
            torch.stack([x2, y2], dim=1),
            torch.stack([x1, y2], dim=1),
        ])

        rotation_matrix = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]
        ], dtype=torch.float32)

        center = self.size / 2
        corners = corners - center
        corners = corners.permute(1, 0, 2)
        rotated = torch.matmul(corners, rotation_matrix.T)
        rotated = rotated + center
        rotated = rotated.permute(1, 0, 2)

        x_min, _ = rotated[:, :, 0].min(dim=0)
        y_min, _ = rotated[:, :, 1].min(dim=0)
        x_max, _ = rotated[:, :, 0].max(dim=0)
        y_max, _ = rotated[:, :, 1].max(dim=0)

        x_min = x_min.clamp(0, self.size)
        y_min = y_min.clamp(0, self.size)
        x_max = x_max.clamp(0, self.size)
        y_max = y_max.clamp(0, self.size)

        cx_new = (x_min + x_max) / 2 / self.size
        cy_new = (y_min + y_max) / 2 / self.size
        w_new = (x_max - x_min) / self.size
        h_new = (y_max - y_min) / self.size

        mask = (w_new > 0) & (h_new > 0)
        target = target[mask]
        target[:, 1] = cx_new[mask]
        target[:, 2] = cy_new[mask]
        target[:, 3] = w_new[mask]
        target[:, 4] = h_new[mask]

        return target

    def crop_resize_bboxes(self, target, left, top, crop_size, final_size):
        cx = target[:, 1] * final_size
        cy = target[:, 2] * final_size
        w = target[:, 3] * final_size
        h = target[:, 4] * final_size

        cx = cx - left
        cy = cy - top

        mask = (cx >= 0) & (cx <= crop_size) & (cy >= 0) & (cy <= crop_size)
        target = target[mask]

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]

        scale = final_size / crop_size
        cx = cx * scale / final_size
        cy = cy * scale / final_size
        w = w * scale / final_size
        h = h * scale / final_size

        target[:, 1] = cx
        target[:, 2] = cy
        target[:, 3] = w
        target[:, 4] = h

        return target
