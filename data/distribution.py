import os
from collections import defaultdict

def count_yolo_classes(label_dir='train/labels'):
    class_counts = defaultdict(int)
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            cls_id = int(line.split()[0])
                            class_counts[cls_id] += 1
    for cls in class_counts:
        print("Class ", cls,": ", class_counts[cls])
    return dict(class_counts)

count_yolo_classes()