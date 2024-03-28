import cv2
import numpy as np
from ultralytics import YOLO

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Define the position for the text annotation on the image
text_position_total = (10, 50)

# Define font, scale, and color for the annotation
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

# Open the video
cap = cv2.VideoCapture('4K Video of Highway Traffic!.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_sample_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize set to store unique vehicle identifiers (e.g., based on bounding box coordinates)
unique_vehicle_ids = set()

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Create a copy of the original frame to modify
        detection_frame = frame.copy()

        # Perform inference on the modified frame
        results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)

        # Retrieve the bounding boxes from the results
        bounding_boxes = results[0].boxes

        # Loop through each bounding box to count unique vehicles
        for box in bounding_boxes.xyxy:
            # Generate a unique identifier for each vehicle based on bounding box coordinates
            vehicle_id = tuple(box.cpu().detach().numpy().astype(int))

            if vehicle_id not in unique_vehicle_ids:
                unique_vehicle_ids.add(vehicle_id)

        # Add a background rectangle for the total unique vehicle count
        cv2.rectangle(processed_frame, (text_position_total[0]-10, text_position_total[1] - 25),
                      (text_position_total[0] + 460, text_position_total[1] + 10), background_color, -1)

        # Add the total unique vehicle count text on top of the rectangle
        cv2.putText(processed_frame, f'Total Unique Vehicles: {len(unique_vehicle_ids)}', text_position_total,
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Display the processed frame
        cv2.imshow('Real-time Traffic Analysis', processed_frame)

        # Write the processed frame to the output video file
        out.write(processed_frame)

        # Press Q on keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and video write objects
cap.release()
out.release()

# Close all the frames
cv2.destroyAllWindows()

# Save the total unique vehicle count to a file
with open('total_unique_vehicle_passes.txt', 'w') as file:
    file.write(str(len(unique_vehicle_ids)))
