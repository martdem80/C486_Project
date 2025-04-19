import sys
import cv2
import sqlite3
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import QTimer, Qt

class WelcomeScreen(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 600, 400)
        self.setStyle()
        self.initUI(start_callback)

    def setStyle(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#2C2F33"))  
        self.setPalette(palette)

    def initUI(self, start_callback):
        title = QLabel("Welcome to Sign Language Recognition", self)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #00CED1;")

        members = QLabel("Patrick Ryan, Woojeong Kim, Devyn Martinez,\nGabe Martinez, and Sicong Tian", self)
        members.setFont(QFont("Arial", 12))
        members.setStyleSheet("color: white;")

        img_label = QLabel(self)
        pixmap = QPixmap("image.png")  
        img_label.setPixmap(pixmap)
        img_label.setScaledContents(True)
        img_label.setFixedSize(400, 200)

        start_btn = QPushButton("START", self)
        start_btn.setStyleSheet("background-color: #00CED1; color: black; font-size: 16px; padding: 10px;")
        start_btn.clicked.connect(start_callback)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(members)
        layout.addWidget(img_label)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)  
        self.setLayout(layout)

class ASLRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 800, 600)
        self.setStyle()
        self.initUI()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setStyle(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#23272A")) 
        self.setPalette(palette)

    def initUI(self):
        title = QLabel("Sign Language Recognition", self)
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setStyleSheet("color: #00CED1;")

        # Camera feed section
        self.image_label = QLabel("Camera Not Started", self)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 2px solid white;")

        # Recognition result section
        self.gesture_label = QLabel("Place your hand in the camera", self)
        self.gesture_label.setFixedSize(200, 150)
        self.gesture_label.setStyleSheet("border: 2px solid white; color: white;")

        # Sign language reference image
        reference_label = QLabel(self)
        ref_pixmap = QPixmap("asl_reference.png")  # Load ASL chart
        reference_label.setPixmap(ref_pixmap)
        reference_label.setScaledContents(True)
        reference_label.setFixedSize(400, 200)

        # Buttons
        self.start_btn = QPushButton("Start Camera", self)
        self.start_btn.setStyleSheet("background-color: #00CED1; color: black; padding: 10px;")
        self.start_btn.clicked.connect(self.start_camera)

        self.exit_btn = QPushButton("Exit", self)
        self.exit_btn.setStyleSheet("background-color: #DC143C; color: white; padding: 10px;")
        self.exit_btn.clicked.connect(self.close)

        # Layouts
        top_layout = QVBoxLayout()
        top_layout.addWidget(title)

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.image_label)
        cam_layout.addWidget(self.gesture_label)

        ref_layout = QVBoxLayout()
        ref_layout.addWidget(reference_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.exit_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(cam_layout)
        main_layout.addLayout(ref_layout)
        main_layout.addLayout(btn_layout)
        
        self.setLayout(main_layout)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qImg))
            self.gesture_label.setText("Detected Gesture: [Placeholder]")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    main_app = ASLRecognitionApp()

    # Define a function to switch screens
    def start_application():
        welcome.close()
        main_app.show()

    welcome = WelcomeScreen(start_application)  
    welcome.show()
    
    sys.exit(app.exec_())
