import cv2              # For video capture and image processing
import mediapipe as mp  # For hand detection and tracking
import torch            # For deep learning framework
import torch.nn as nn   # For neural network layers and operations
import numpy as np      # For array operations and data manipulation
from torchvision import transforms  # For image transformations
# Updated import 
from torchvision.models import mobilenet_v3_large   # For loading the MobileNetV3-Large pre-trained model    # MODIFIED
from PIL import Image   # For handling image input/output and format conversion     #MODIFIED

class RealTimeRecognizer:
    """
    Provides real-time ASL recognition by capturing video frames, detecting hands with MediaPipe,
    and classifying sign gestures using a pre-trained MobileNetV3 model.
    """
    # Updated default num_classes to 25  # MODIFIED
    def __init__(self, model_path, num_classes = 25, camera_index = 0):     # MODIFIED
        self.model = self.load_model(model_path, num_classes)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode = False,    # Live video feed mode
            max_num_hands = 1,
            min_detection_confidence = 0.7
        )

        self.mp_draw = mp.solutions.drawing_utils   # Utility to draw hand landmarks.
        self.cap = cv2.VideoCapture(camera_index)
        # Updated ASL map to include extra class '_'  # MODIFIED
        self.asl_map = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']] 

        if num_classes == 25:   # MODIFIED
            self.asl_map.append('_')  # Extra class for non-letter or space



    def load_model(self, model_path, num_classes = 25):
        """Load and configure the pre-trained MobileNetV3-Small model."""
        # Use MobileNetV3-Large with its default weights  # MODIFIED
        model = mobilenet_v3_large()    # Updated to use MobileNetV3-Large  # MODIFIED
        
        # Updated classifier architecture   # MODIFIED
        model.classifier = nn.Sequential(
            nn.Linear(960, 1024),   # Changed from 576 to 960 for MobileNetV3-Large
            nn.Hardswish(),
            nn.Dropout(p=0.4),      # MODIFIED
            nn.Linear(1024, 512),   # MODIFIED 
            nn.Hardswish(),         # MODIFIED 
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)    # MODIFIED 
        )

        # Load the checkpoint and apply the saved weights.
        checkpoint = torch.load(model_path, map_location = torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()    # Set the model to evaluation mode.
        return model


    def preprocess_hand(self, roi):
        """Preprocess the hand region of interest (ROI) for model prediction."""
        # Resize directly to 50x50 while preserving color information
        # roi_resized = cv2.resize(roi, (224, 224)) # COMMENTED
        roi_resized = cv2.resize(roi, (50, 50))     # MODIFIED
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image  # MODIFIED
        pil_image = Image.fromarray(roi_rgb)
        
        # Use the transform pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),      # Convert the PIL to a Tensor   # MODIFIED
            transforms.Resize(224),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

        # Apply transformations to the PIL image and add batch  # MODIFIED
        roi_tensor = transform(pil_image)
        roi_tensor = roi_tensor.unsqueeze(0)

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
                
                # Dynamic padding with an adjustable ratio for ROI extraction.
                # Set the padding ratio for 10% for now, but it can be adjusted.
                pad_ratio = 0.1     # Adjustable value for better results
               
                box_width = x_max - x_min
                box_height = y_max - y_min
                pad_x = int(pad_ratio * box_width)   # 10% of the box's width
                pad_y = int(pad_ratio * box_height)  # 10% of the box's height

                x_min = max(0, x_min - pad_x)
                y_min = max(0, y_min - pad_y)
                x_max = min(w, x_max + pad_x)
                y_max = min(h, y_max + pad_y)
             
                # Crop the hand region (ROI)
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    preprocessed_roi = self.preprocess_hand(roi)
                    prediction_idx = self.predict(preprocessed_roi)
                    
                    # Handle prediction based on index  # MODIFIED
                    if prediction_idx < len(self.asl_map):
                        prediction_letter = self.asl_map[prediction_idx]
                    else:
                        prediction_letter = "Unknown" 

                    prediction_text = f"Prediction: {prediction_letter}"    # MODIFIED
                
                # Draw the bounding box
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
    recognizer = RealTimeRecognizer('./model_training/best_model_v2.pth', num_classes = 25)   # MODIFIED
    recognizer.run()
