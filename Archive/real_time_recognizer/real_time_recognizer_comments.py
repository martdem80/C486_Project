import cv2                                          # OpenCV: For video capture and image processing
import mediapipe as mp                              # MediaPipe: For real-time hand detection and tracking
import torch                                        # PyTorch: Deep learning framework
import torch.nn as nn                               # PyTorch's module for neural network layers and operations
import numpy as np                                  # NumPy: For array operations and numerical processing
from torchvision import transforms                  # For image transformation pipelines
from torchvision.models import mobilenet_v3_small   # Pre-trained MobileNetV3 model for transfer learning

class RealTimeRecognizer:
    """
    Aim: RealTimeRecognizer enables real-time ASL recognition.
         
    Enables real-time ASL recognition by capturing video with cv2,
    detecting hands using MediaPipe,
    and classifying gestures with a pre-trained MobileNetV3-Small model.
    Displays predictions in real-time on the video feed via cv2.

    """
    
    def __init__(self, model_path, num_classes = 24, camera_index = 0):
        """
        Aim: Initialize the RealTimeRecognizer.
        
        Parameters:
        - model_path (str): Path to the pre-trained model checkpoint.
        - num_classes (int): Number of ASL gesture classes (default 24, representing letters A-Y excluding J and Z).
        - camera_index (int): The index of the camera to be used (default 0 for the primary camera).
    
        Setup includes:
        - Loading the deep learning model.
        - Initializing MediaPipe for hand detection.
        - Setting up the drawing utilities to visualize landmarks.
        - Starting the video capture.
        - Creating an ASL mapping for predictions.
        """
        # Load the MobileNetV3 model with custom classifier for ASL recognition.
        self.model = self.load_model(model_path, num_classes)
        
        # Initialize MediaPipe Hands for real-time hand detection.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode = False,     # Process a continuous video stream.
            max_num_hands = 1,             # Detect only one hand to simplify processing.
            min_detection_confidence=0.7 # Minimum confidence threshold for hand detection.
        )
        # Utility to draw detected hand landmarks on the frame.
        self.mp_draw = mp.solutions.drawing_utils
        
        # Start video capture from the specified camera.
        self.cap = cv2.VideoCapture(camera_index)
        
        # Create a mapping for ASL letters (A to Z excluding 'J' and 'Z' which require motion).
        self.asl_map = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
    
    
    def load_model(self, model_path, num_classes = 24):
        """
        Aim: Load and configure the pre-trained MobileNetV3-Small model for ASL recognition.
        
        Parameters:
        - model_path (str): Path to the checkpoint file containing saved weights.
        - num_classes (int): Number of ASL classes.
        
        Returns:
        - model (torch.nn.Module): The MobileNetV3 model set to evaluation mode.
        
        Steps:
        1. Instantiate the MobileNetV3-Small model.
        2. Replace its classifier with a custom fully connected network.
        3. Load the saved state dictionary from the checkpoint.
        4. Set the model to evaluation mode (disables dropout, batch norm updates).
        """
        model = mobilenet_v3_small()
        model.classifier = nn.Sequential(
            nn.Linear(576, 1024),  # Expands feature dimension to capture more nuanced patterns.
            nn.Hardswish(),        # Activation function recommended in MobileNetV3 for non-linearity.
            nn.Dropout(p = 0.3),      # Dropout to reduce overfitting.
            nn.Linear(1024, num_classes)  # Final layer mapping features to ASL gesture classes.
        )
    
        # Load the checkpoint containing the trained weights.
        checkpoint = torch.load(model_path, map_location = torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set model to evaluation mode to disable training-specific layers.
        return model

    def preprocess_hand(self, roi):
        """
        Aim: Preprocess the hand region of interest (ROI) to prepare it for model prediction.
        
        Parameters:
        - roi (numpy.ndarray): Cropped image containing the hand.
        
        Returns:
        - roi_tensor (torch.Tensor): Preprocessed image tensor ready for inference.
        
        Processing steps:
        1. Convert the ROI to grayscale (simplifies input and matches training conditions).
        2. Resize the image to 28x28 pixels (as used in model training).
        3. Normalize pixel values to the [0, 1] range.
        4. Duplicate the single grayscale channel to form a 3-channel image (required by MobileNetV3).
        5. Convert the image into a PyTorch tensor.
        6. Define and apply a transformation that:
           - Resizes the tensor to 224x224 pixels.
           - Normalizes the image using ImageNet statistics.
        7. Unsqueeze the tensor to add a batch dimension.
        """
        # Convert the hand region from BGR to grayscale.
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Resize to 28x28 pixels to match training image dimensions.
        roi_resized = cv2.resize(roi_gray, (28, 28))
        # Normalize the pixel values to be between 0 and 1.
        roi_normalized = roi_resized.astype(np.float32) / 255.0
        # Duplicate the grayscale image to create 3 identical channels.
        roi_3channel = np.stack([roi_normalized] * 3, axis=0)
        # Convert the 3-channel image into a torch tensor.
        roi_tensor = torch.FloatTensor(roi_3channel)
        # Define transformation: resize to 224x224 and normalize using ImageNet mean and std.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet standard deviation
            )
        ])
        # Add a batch dimension and apply the transformation.
        roi_tensor = transform(roi_tensor.unsqueeze(0))
        return roi_tensor

    def predict(self, image_tensor):
        """
        Aim: Perform model inference on the preprocessed image tensor to predict the ASL gesture.
        
        Parameters:
        - image_tensor (torch.Tensor): The image tensor processed from the ROI.
        
        Returns:
        - prediction (int): Index corresponding to the predicted ASL gesture.
        
        Inference is done without computing gradients to optimize performance.
        """
        with torch.no_grad():
            output = self.model(image_tensor)  # Forward pass through the model.
            prediction = output.argmax(dim = 1).item()  # Get the index with the highest score.
        return prediction

    def process_frame(self, frame):
        """
        Aim: Process a single video frame to detect a hand, extract its ROI, and predict the ASL gesture.
        
        Parameters:
        - frame (numpy.ndarray): The captured video frame in BGR format.
        
        Returns:
        - frame (numpy.ndarray): The processed frame with hand landmarks, bounding box, and prediction text overlay.
        
        Steps:
        1. Convert the frame from BGR to RGB as required by MediaPipe.
        2. Use MediaPipe to detect hand landmarks.
        3. If landmarks are found:
           a. Draw the landmarks and connections on the frame.
           b. Compute a bounding box around the hand by finding minimum and maximum landmark coordinates.
           c. Add padding to the bounding box for better ROI capture.
           d. Crop the ROI from the frame.
           e. Preprocess the ROI and perform prediction.
           f. Update the prediction text with the corresponding ASL letter.
           g. Draw the bounding box around the hand.
        4. Overlay the prediction text on the frame.
        """
        # Convert frame from BGR to RGB for MediaPipe processing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        prediction_text = "Detecting..."
        
        if results.multi_hand_landmarks:
            # Process each detected hand (in our case, typically one hand).
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections on the frame.
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Retrieve the frame dimensions.
                h, w, _ = frame.shape
                # Initialize bounding box coordinates.
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                # Loop through all landmarks to determine the bounding box.
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Define padding to include extra area around the hand.
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Crop the hand region of interest (ROI) from the frame.
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    # Preprocess the ROI and run the prediction.
                    preprocessed_roi = self.preprocess_hand(roi)
                    prediction_idx = self.predict(preprocessed_roi)
                    # Map the predicted index to the corresponding ASL letter.
                    prediction_text = f"Prediction: {self.asl_map[prediction_idx]}"
                
                # Draw the bounding box around the detected hand.
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Overlay the prediction text on the frame.
        cv2.putText(frame, prediction_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def run(self):
        """
        Aim: Run the real-time ASL recognition loop.
        
        This method continuously captures frames from the camera, processes each frame for hand
        detection and gesture prediction, and displays the annotated video. The loop exits when
        the user presses the 'q' key.
        
        Steps:
        1. Capture frames from the video stream.
        2. Process each frame using process_frame().
        3. Display the processed frame.
        4. Break the loop on a 'q' key press.
        5. Release the video capture and close all display windows.
        """
        while True:
            ret, frame = self.cap.read()  # Read a frame from the camera.
            if not ret:
                break  # If frame reading fails, exit the loop.
            # Process the frame for hand detection and ASL prediction.
            frame = self.process_frame(frame)
            cv2.imshow('ASL Recognition', frame)  # Display the frame in a window.

            # Exit the loop if the 'q' key is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Release the video capture object and close all OpenCV windows.
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create an instance of RealTimeRecognizer using the provided model checkpoint.
    recognizer = RealTimeRecognizer('./model_training/best_train_model.pth')
    # Start the real-time recognition process.
    recognizer.run()
