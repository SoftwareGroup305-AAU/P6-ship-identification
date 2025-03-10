import pandas as pd

def load_label_yolov8(filename):
    data = pd.read_csv(filename, delim_whitespace=True, header=None)
    data.columns = ["class", "center_x", "center_y", "width", "height"]
    bounding_boxes = []
    for index, row in data.iterrows():
        bounding_boxes.append([int(row["class"]), 
                               float(row["center_x"]), 
                               float(row["center_y"]), 
                               float(row["width"]), 
                               float(row["height"])])
    return bounding_boxes