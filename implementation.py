import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import Text, Menu, filedialog
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import os
import mediapipe as mp      

"""
Process: MediaPipe-Based Verification for YOLO Detection Filtering
- Uses MediaPipe to detect hand landmarks.
- For each YOLO detection, checks if at least 30% of landmarks are within the YOLO bounding box.
- Only if the verification passes, the YOLO detection is accepted.
"""

#Global variables
model_path = r".\best.pt"
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

#Checks if camera opened successfully
if not cap.isOpened():
    print("Unable to open camera")
    exit()

last_detected_letter = None #Tracks the last detected letter and saved file path
last_saved_file = None  #Tracks the last saved file

# Initialize MediaPipe 
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode = False,
                                max_num_hands = 1,  # Only detect 1 hand to reduce False Positive
                                min_detection_confidence = 0.5,
                                min_tracking_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

# Processes the frame using MediaPipe to detect hand landmarks first.
def get_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    hands_landmarks = []
    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))
            hands_landmarks.append(landmarks)
    return hands_landmarks

# Checks if at least 30% of landmarks are within the bounding box.
def is_hand_in_box(landmarks, box_coords, threshold = 0.3):
    x1, y1, x2, y2 = box_coords
    landmarks_in_box = sum(1 for (lm_x, lm_y) in landmarks 
                           if x1 <= lm_x <= x2 and y1 <= lm_y <= y2)
    return (landmarks_in_box / len(landmarks)) >= threshold if landmarks else False

def update_frame():
    global cap, model, last_detected_letter

    ret, frame = cap.read()

    if ret:
        # Get hand landmarks from the frame
        hand_landmarks_list = get_hand_landmarks(frame)     

        # Use YOLOv8 model for prediction
        results = model(frame, conf=0.44)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for result in results[0]:
                boxes = result.boxes
                for box in boxes:
                    #Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    #Get confidence
                    confidence = box.conf[0].cpu().numpy()

                    #Get class
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]

                    # Verify that sufficient hand landmarks fall inside the YOLO box.
                    hand_detected = any(is_hand_in_box(landmarks, (x1, y1, x2, y2))
                                        for landmarks in hand_landmarks_list)
                    if not hand_detected:
                        continue  # Skip this detection if verification fails.

                    #Checks if the detected letter is different from the last one
                    if class_name != last_detected_letter:
                        last_detected_letter = class_name
                        text_box.insert(tk.END, class_name + " ")  #Appends letter to the text box
                        text_box.see(tk.END)  #Scrolls to the end of the text box

                    #Draws bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    #Displays class and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        #Converts frame to an image format for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        video_label.imgtk = img
        video_label.configure(image=img)

    #Schedules the next frame update
    video_label.after(10, update_frame)

def delete_last_character():
    current_content = text_box.get("1.0", tk.END).strip()
    if current_content:  #Checks if there's any text
        updated_content = current_content[:-1]
        text_box.delete("1.0", tk.END)
        text_box.insert("1.0", updated_content)

#Menu functions
#Overwrites the last saved file
def save():
    global last_saved_file
    if last_saved_file:
        with open(last_saved_file, "w") as file:
            content = text_box.get("1.0", tk.END).strip()
            file.write(content)
        print(f"File overwritten: {last_saved_file}")
    else:
        print("No file to overwrite. Use 'Save As' first.")

#Saves the content of the text box to the file
def save_as():
    global last_saved_file
    logs_dir = "./LOGS"
    if not os.path.exists(logs_dir): #Ensures the LOGS directory exists
        os.makedirs(logs_dir)

    #Prompts user for file name
    file_name = filedialog.asksaveasfilename(
        initialdir=logs_dir,
        title="Save As",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        defaultextension=".txt"
    )

    if file_name:
        #Saves the file path for future overwrites
        last_saved_file = file_name

        #Saves the content of the text box to the file
        with open(file_name, "w") as file:
            content = text_box.get("1.0", tk.END).strip()
            file.write(content)
        print(f"File saved: {file_name}")

#Prompts user to select a file to open, reads the file contents, replaces textbox text with the file text
def open_file():
    
    file_name = filedialog.askopenfilename(
        title="Open File",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    if file_name:
        try:
            with open(file_name, "r") as file:
                content = file.read()
            text_box.delete("1.0", tk.END)
            text_box.insert("1.0", content)
            print(f"File opened: {file_name}")
        except Exception as e:
            print(f"Failed to open file: {e}")

#Clears all text in the text box
def delete():
    text_box.delete("1.0", tk.END)

def about():
    readme_path = "./README.txt"  #Path to the README file
    if os.path.exists(readme_path):
        try:
            with open(readme_path, "r") as file:
                content = file.read()
            messagebox.showinfo("About", content)
        except Exception as e:
            print(f"Failed to read README.txt file: {e}")
    else:
        print("README.txt file not found.")
        messagebox.showerror("Error", "README.txt file not found.")


#Opens the LOGS folder in the file explorer
def open_folder():
    logs_dir = "LOGS"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    os.startfile(logs_dir)  

def open_references():
    # Path to the image file
    image_path = "./ASL_map.png"  # Replace with your image file path
    if os.path.exists(image_path):
        try:
            # Create a new window for displaying the image
            references_window = tk.Toplevel(app)
            references_window.title("References")

            # Open and display the image using PIL
            image = Image.open(image_path)
            image = image.resize((600, 400))  # Resize the image to fit the window
            img = ImageTk.PhotoImage(image)

            # Add the image to the new window
            image_label = tk.Label(references_window, image=img)
            image_label.image = img  # Keep a reference to avoid garbage collection
            image_label.pack()
        except Exception as e:
            print(f"Failed to open image: {e}")
            messagebox.showerror("Error", f"Failed to open image: {e}")
    else:
        print("Image file not found.")
        messagebox.showerror("Error", "Image file not found.")

# Initialize main application window
app = tk.Tk()
app.title("Gesture Recognition")
app.geometry("800x600")

# Create a drop-down menu
menu_bar = Menu(app)
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Save", command=save)
file_menu.add_command(label="Save As", command=save_as)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Delete", command=delete)
file_menu.add_command(label="About", command=about)
file_menu.add_command(label="Open Folder", command=open_folder)
menu_bar.add_cascade(label="Menu", menu=file_menu)

# Configure the application to use the menu
app.config(menu=menu_bar)

# Camera feed label
video_label = tk.Label(app)
video_label.pack()

# Text box for displaying detected letters
text_box = Text(app, height=5, width=40)
text_box.pack()

# Button to delete the last character
delete_button = tk.Button(app, text="Delete Last Character", command=delete_last_character)
delete_button.pack()

#add 'references' button to GUI
references_button = tk.Button(app, text="References", command=open_references)
references_button.pack()

# Start the frame update
update_frame()

# Run the Tkinter application loop
app.mainloop()

# Release resources after closing the GUI
cap.release()
cv2.destroyAllWindows()