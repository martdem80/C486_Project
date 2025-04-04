import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time


def main():
    # Load YOLOv8 model
    model_path = r"C:\Users\tians\OneDrive\Desktop\yolov8\best.pt"
    model = YOLO(model_path)

    # Open camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Unable to open camera")
        return

    # Set up window
    cv2.namedWindow("ASL Recognition", cv2.WINDOW_NORMAL)

    while True:
        # Read camera frame
        ret, frame = cap.read()

        if not ret:
            print("Unable to receive frame (video stream may have ended)")
            break

        # Use YOLOv8 model for prediction
        results = model(frame, conf=0.5)  # conf is confidence threshold

        # Process results
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for result in results[0]:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Get confidence
                    confidence = box.conf[0].cpu().numpy()

                    # Get class
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display class and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show results
        cv2.imshow("Gesture Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()