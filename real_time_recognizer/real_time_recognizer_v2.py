import cv2              # For video capture and image processing
import mediapipe as mp  # For hand detection and tracking
import torch            # For deep learning framework
import torch.nn as nn   # For neural network layers and operations
import numpy as np      # For array operations and data manipulation
from torchvision import transforms  # For image transformations
# Updated import: using MobileNetV3-Large and its weights  # MODIFIED
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights  # MODIFIED

class RealTimeRecognizer:
    """
    Provides real-time ASL recognition by capturing video frames, detecting hands with MediaPipe,
    and classifying sign gestures using a pre-trained MobileNetV3 model.
    """
    # Updated default num_classes to 25  # MODIFIED
    def __init__(self, model_path, num_classes = 25, camera_index = 0):     # MODIFIED
        self.model = self.load_model(model_path, num_classes)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode = False,    # Live video feed mode.
            max_num_hands = 1,
            min_detection_confidence = 0.7
        )

        self.mp_draw = mp.solutions.drawing_utils   # Utility to draw hand landmarks.
        self.cap = cv2.VideoCapture(camera_index)
        # Updated ASL map to include extra class '_'  # MODIFIED
        self.asl_map = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']] + ['_']  # MODIFIED'Z']]
    

    def load_model(self, model_path, num_classes = 25):
        """Load and configure the pre-trained MobileNetV3-Small model."""
        # Use MobileNetV3-Large with its default weights  # MODIFIED
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)  # MODIFIED
        
        # Updated classifier architecture to match model_training_v2.py  # MODIFIED
        model.classifier = nn.Sequential(
            nn.Linear(960, 1024),   # Changed from 576 to 960 for MobileNetV3-Large
            nn.Hardswish(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # Load the checkpoint and apply the saved weights.
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()    # Set the model to evaluation mode.
        return model


    def preprocess_hand(self, roi):
        """Preprocess the hand region of interest (ROI) for model prediction."""
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (28, 28))
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        roi_3channel = np.stack([roi_normalized] * 3, axis=0)
        roi_tensor = torch.FloatTensor(roi_3channel)
        
        # Define a transform to resize to 224x224 and normalize using ImageNet stats
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

        # Apply the transformation. Unsqueeze to add the batch dimension.
        roi_tensor = transform(roi_tensor.unsqueeze(0))
        return roi_tensor


    def predict(self, image_tensor):
        """Run inference on the preprocessed image tensor."""
        # Disable gradient computation for inference.
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.argmax(dim = 1).item()
        return prediction


    def process_frame(self, frame):
        """Process a single video frame for hand detection and ASL prediction."""
        # Convert the frame from BGR to RGB (required by MediaPipe).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        prediction_text = "Detecting..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Extract bounding box around the hand
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add padding to the bounding box
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Crop the hand region (ROI)
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    preprocessed_roi = self.preprocess_hand(roi)
                    prediction_idx = self.predict(preprocessed_roi)
                    prediction_text = f"Prediction: {self.asl_map[prediction_idx]}"
                
                # Draw the bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display prediction text on the frame
        cv2.putText(frame, prediction_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame


    def run(self):
        """Main loop to capture video frames, process them, and display predictions."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Process the current frame for hand detection and prediction
            frame = self.process_frame(frame)
            cv2.imshow('ASL Recognition', frame)

            # Break the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Updated model path to best_model_v2.pth  # MODIFIED
    recognizer = RealTimeRecognizer('./model_training/best_model_v2.pth')   # MODIFIED
    recognizer.run()
