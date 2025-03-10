"""
Real-Time ASL Recognition with Detailed Comments.
This code captures live video from a webcam, detects hand landmarks using MediaPipe,
extracts the hand Region Of Interest (ROI), preprocesses the ROI for a pre-trained MobileNetV3 model,
and predicts the corresponding ASL letter. The prediction is displayed in real-time on the video feed.
"""

import cv2                                          # OpenCV: For video capture and image processing
import mediapipe as mp                              # MediaPipe: For real-time hand detection and tracking
import torch                                        # PyTorch: Deep learning framework
import torch.nn as nn                               # PyTorch's module for neural network layers and operations
import numpy as np                                  # NumPy: For array operations and numerical processing
from torchvision import transforms                  # For image transformation pipelines
from torchvision.models import mobilenet_v3_small   # Pre-trained MobileNetV3 model for transfer learning


def load_model(model_path, num_classes=24):
    """
    Aim: Load the pre-trained MobileNetV3 model with a custom classifier suited for ASL recognition.

    Parameters:
        model_path (str): Path to the saved model checkpoint.
        num_classes (int): The number of ASL classes (default is 24).

    Returns: model (torch.nn.Module): The model loaded with trained weights, set to evaluation mode.
    """
    # Initialize the pre-trained MobileNetV3 model
    model = mobilenet_v3_small()
    
    # Replace the original classifier with a custom one for our task:
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),         # Expand feature dimension from 576 to 1024
        nn.Hardswish(),               # Activation function as recommended in MobileNetV3
        nn.Dropout(p=0.3),            # Dropout for regularization
        nn.Linear(1024, num_classes)  # Final classification layer mapping features to ASL classes
    )

    # Load model weights from the checkpoint; map to CPU for compatibility
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_hand(roi):
    """
    Aim: Preprocess the region of interest (ROI) containing the hand to prepare it for model prediction.

    Steps:
        1. Resize the ROI to 224x224 pixels.
        2. Convert the ROI to a tensor.
        3. Normalize the tensor using ImageNet statistics.

    Parameters: roi (numpy.ndarray): The cropped image region in BGR format (as obtained from OpenCV).

    Returns: roi_tensor (torch.Tensor): The preprocessed image tensor with an added batch dimension.
    """
    # Define the transformation pipeline: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to the size expected by MobileNetV3
        transforms.ToTensor(),          # Convert PIL Image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet's mean
                             std=[0.229, 0.224, 0.225])   # Normalize using ImageNet's std
    ])
    
    # Convert the image from BGR (OpenCV) to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # Convert the numpy array image to a PIL Image
    roi_pil = transforms.ToPILImage()(roi_rgb)
    # Apply the defined transformation and add a batch dimension
    roi_tensor = transform(roi_pil).unsqueeze(0)
    return roi_tensor

def predict(model, image_tensor):
    """
    Aim: 
        Predict the ASL letter from the preprocessed image tensor.

    Parameters:
        model (torch.nn.Module): The pre-trained ASL recognition model.
        image_tensor (torch.Tensor): The preprocessed hand ROI tensor.

    Returns:
        prediction (int): The index corresponding to the predicted ASL class.
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(image_tensor)  # Forward pass through the model
        prediction = output.argmax(dim=1).item()  # Get the class with the highest probability
    return prediction


# Real-Time Hand Tracking Setup using MediaPipe
# Initialize MediaPipe Hands module for detecting hand landmarks in real time
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # Use dynamic mode for video stream
    max_num_hands=1,                # Process at most one hand per frame
    min_detection_confidence=0.7    # Minimum confidence threshold for detections
)
mp_draw = mp.solutions.drawing_utils  # Utility to draw landmarks on images


# Initialize Video Capture and Load the Model
cap = cv2.VideoCapture(0)  # Start capturing video from the default camera (webcam)


# Load the pre-trained model from the specified checkpoint path
model = load_model('./model_training/best_train_model.pth')


# Define a mapping from the model output indices to ASL letters.
# ASL alphabets: A to Z, excluding 'J' and 'Z' (which involve motion)
asl_map = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]


# Main Loop: Process Each Video Frame in Real Time
while True:
    ret, frame = cap.read()  # Capture a frame from the video stream
    if not ret:
        break  # Exit loop if frame capture fails

    # Convert the captured frame from BGR (OpenCV default) to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame to detect hand landmarks
    results = hands.process(rgb_frame)

    # Initialize the text to display on the frame
    prediction_text = "Detecting..."

    # If hand landmarks are detected in the frame:
    if results.multi_hand_landmarks:
        # Process each detected hand (here, limited to one hand)
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Compute the bounding box for the detected hand
            h, w, _ = frame.shape   # Get the frame dimensions
            x_min, y_min = w, h     # Initialize min coordinates to max possible values
            x_max, y_max = 0, 0     # Initialize max coordinates to zero

            # Loop through each landmark point to determine the bounding box
            for lm in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add Padding to the Bounding Box
            padding = 20    # Add extra space around the detected hand
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Extract and Process the Hand ROI
            roi = frame[y_min:y_max, x_min:x_max]  # Crop the hand region from the frame

            if roi.size > 0:
                preprocessed_roi = preprocess_hand(roi)  # Preprocess the ROI for the model
                prediction_idx = predict(model, preprocessed_roi)  # Get the model prediction
                # Map the prediction index to the corresponding ASL letter
                prediction_text = f"Prediction: {asl_map[prediction_idx]}"

            # Draw a rectangle around the detected hand for visualization
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the Result on the Frame
    cv2.putText(frame, prediction_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Overlay prediction text
    cv2.imshow('ASL Recognition', frame)  # Show the video frame with annotations

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture and close all OpenCV windows after the loop ends
cap.release()
cv2.destroyAllWindows()
