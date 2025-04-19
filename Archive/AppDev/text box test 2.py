from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
import sys

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Main window
        self.setWindowTitle("Center Text Box Example")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # Text box
        self.textbox = QLineEdit(self)
        self.textbox.setAlignment(Qt.AlignCenter) 

        layout.addWidget(self.textbox)

        # Save button
        save_button = QPushButton("Save Text")
        save_button.clicked.connect(self.save_text)

        layout.addWidget(save_button)

      
        self.setLayout(layout)

    def save_text(self):
     
        text = self.textbox.text()
        
        # Open a file dialog to choose the save location
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Text", "", "Text Files (*.txt);;All Files (*)")
        
        if file_name:
            # Save the text to a file
            with open(file_name, "w") as file:
                file.write(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
