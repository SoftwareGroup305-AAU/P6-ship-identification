import os

old_classes = {
    5: 1  # Fish-Boat → 1 (Fishing boat)
}

def relabel_yolo(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except PermissionError:
        print(f"Permission denied: {file_path}")
        return
    except Exception as e:
        print(f"Unexpected error with file {file_path}: {e}")
        return

    new_lines = []
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        new_class = old_classes.get(class_id, 0)  # Default to "Not fishing boat"
        new_lines.append(f"{new_class} " + " ".join(parts[1:]) + "\n")

    # Add whitespace at the end of lines where the class is "1"
    if any(line.startswith("1 ") for line in new_lines):  # Check if there's any line with class "1"
        new_lines = [line.rstrip() + " " if line.startswith("1 ") else line for line in new_lines]

    try:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Successfully processed: {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

# Apply to all labels
label_dir = "Add label here"

# List files in the directory and check if they are being processed
for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(label_dir, file)
        relabel_yolo(file_path)
