from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog
from PyQt6.QtCore import Qt
import sys

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        #main window
        self.setWindowTitle("Center Text Box Example")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        #text box
        self.textbox = QLineEdit(self)
        self.textbox.setAlignment(Qt.AlignmentFlag.AlignCenter) 

        layout.addWidget(self.textbox)

        #save button
        save_button = QPushButton("Save Text")
        save_button.clicked.connect(self.save_text)

        layout.addWidget(save_button)

        #layout for the main window
        self.setLayout(layout)

    def save_text(self):
        #get the text from the text box
        text = self.textbox.text()
        
        #open a file dialog to choose the save location
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Text", "", "Text Files (*.txt);;All Files (*)")
        
        if file_name:
            #save the text to a file
            with open(file_name, "w") as file:
                file.write(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
