import cv2  # For video capture and image processing
import mediapipe as mp  # For hand detection and tracking
import torch  # For deep learning framework
import torch.nn as nn  # For neural network layers and operations
import numpy as np  # For array operations and data manipulation
from torchvision import transforms  # For image transformations
from torchvision.models import mobilenet_v3_small  # For pre-trained MobileNetV3 model

# Load trained model
def load_model(model_path, num_classes=24):
    model = mobilenet_v3_small()
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, num_classes)
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# Preprocess hand ROI 
def preprocess_hand(roi):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_pil = transforms.ToPILImage()(roi_rgb)
    roi_tensor = transform(roi_pil).unsqueeze(0)
    return roi_tensor


# Predict ASL letter 
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()
    return prediction


# Real-Time hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize camera 
cap = cv2.VideoCapture(0)

# Load the pre-trained model 
model = load_model('./model_training/best_train_model.pth')

# Mapping index to ASL letters (A-Y, excluding J)
asl_map = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = "Detecting..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop hand ROI
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                preprocessed_roi = preprocess_hand(roi)
                prediction_idx = predict(model, preprocessed_roi)
                prediction_text = f"Prediction: {asl_map[prediction_idx]}"

            # Draw rectangle around hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display prediction text
    cv2.putText(frame, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show camera feed
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
