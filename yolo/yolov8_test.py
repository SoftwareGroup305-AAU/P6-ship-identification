from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Path to the image
image_path = r"C:\dev\python\P6-ship-identification\yolo\Bottom-trawler-shutterstock_1955173456-scaled-1.jpg"

# Path to the trained model
best_path = r"C:\dev\python\P6-ship-identification\yolo\best.pt"

# Load the YOLO model
model = YOLO(best_path)

# Perform prediction
results = model(image_path)  # You can also use model.predict(image_path)

# Extract the first result (assuming single image input)
result = results[0]

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (Matplotlib)

# Draw bounding boxes on the image
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
    confidence = box.conf[0].item()  # Confidence score
    cls = int(box.cls[0].item())  # Class index
    
    # Draw rectangle and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue bounding box
    cv2.putText(image, f"Class '{result.names[cls]}' ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with bounding boxes
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")  # Hide axes
plt.show()

