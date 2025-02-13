from PyQt5.QtWidgets import QApplication, QWidget 

import sys

#pass in sys.argv to allow command line arguments for app
app = QApplication(sys.argv)

#create a Qt widget, which will be our window
window = QWidget()
window.show()

app.exec()
