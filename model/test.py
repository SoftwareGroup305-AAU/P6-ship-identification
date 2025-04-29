from torchvision.io import read_image
from torchvision import transforms
from core import YOLO
import torch

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

model_file = r"yolo_custom_best.pth"

test_img = r"yolo/data/test/images/images.jpeg"

img = read_image(test_img)

img = train_transforms(img).unsqueeze(0)

model = YOLO(num_classes=11)

model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device("cpu")))

model.eval()

class_pred, obj_pred, localization_pred = model(img)

for grid_y in range(localization_pred[0].shape[0]):
    for grid_x in range(localization_pred[0][0].shape[0]):
            print(localization_pred[0][grid_x][grid_y])

            



