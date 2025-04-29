import glob
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_dir = "yolo/data/train/images/"
label_dir = "yolo/data/train/labels/"

# Get all image files (you can filter by pattern like *.jpg or *.png)
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

for image_path in image_paths:
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_path):
        print(f"[!] No label found for {base}")
        continue

    # Load image
    img = read_image(image_path).float() / 255.0
    img = img.permute(1, 2, 0).numpy()
    img_h, img_w = img.shape[:2]

    # Load labels
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f.readlines()]

    # Plot image + boxes
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for label in labels:
        class_id, x_c, y_c, w, h = label
        x_c *= img_w
        y_c *= img_h
        w *= img_w
        h *= img_h
        x_min = x_c - w / 2
        y_min = y_c - h / 2

        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"GT: {int(class_id)}", color='lime', fontsize=8, backgroundcolor='black')

    plt.title(base)
    plt.axis('off')
    plt.show()
