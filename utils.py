import json
import os

def load_class_dict(classes_path):
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as file:
            return json.load(file)
    return {}


def load_class_array(classes_path):
    classes = load_class_dict(classes_path)
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result