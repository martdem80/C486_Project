from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASL interpreter")

        button = QPushButton("Press here!")
        
        self.setCentralWidget(button)

        self.setMinimumSize(QSize(400,300))



#pass in sys.argv to allow command line arguments for application
app = QApplication(sys.argv)

#create a Qt widget, which will be our window (in Qt all top level widgets are windows)
window = MainWindow()
window.show()

app.exec()
