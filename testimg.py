import cv2
from ultralytics import YOLO

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Read the image
image_path = 'sam2.jpg'
image = cv2.imread(image_path)

# Perform inference on the image
results = best_model.predict(image, imgsz=640, conf=0.6)

# Retrieve the bounding boxes from the results
bounding_boxes = results[0].boxes

# Count the total number of vehicles
total_vehicles = len(bounding_boxes.xyxy)

# Display the total count on the image
annotated_image = results[0].plot(line_width=2)
cv2.putText(annotated_image, f'Total Vehicles: {total_vehicles}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image with bounding boxes and total count
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()