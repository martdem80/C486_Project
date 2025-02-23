from PyQt5.QtCore import QSize, Qt, QProcess, QRect
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont 
from PyQt5.QtWidgets import(
    QApplication, 
    QWidget, 
    QMainWindow, 
    QPushButton, 
    QMainWindow,
    QPushButton,
    QToolBar,
    QLabel,
    QLineEdit,
    QMenu,
    QMenuBar,
    QStatusBar,
    QAction, 
    QVBoxLayout, QHBoxLayout,
    QListView,

)
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASL interpreter")

      #  button = QPushButton("Press here!")

        self.actionSave_as = QAction(MainWindow)
        self.actionSave_as.setObjectName("actionSave_as")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.hello_label = QLabel(self.centralwidget)
        self.hello_label.setObjectName("hello_label")
        self.hello_label.setGeometry(QRect(130, 10, 421, 41))
        font = QFont()
        font.setItalic(True)
        self.hello_label.setFont(font)
        self.image_label = QLabel(self.centralwidget)
        self.image_label.setObjectName("image_label")
        self.image_label.setGeometry(QRect(130, 120, 571, 311))
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 106, 431))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.newdraft = QPushButton(self.verticalLayoutWidget)
        self.newdraft.setObjectName("newdraft")

        self.verticalLayout.addWidget(self.newdraft)

        self.moch1 = QPushButton(self.verticalLayoutWidget)
        self.moch1.setObjectName("moch1")

        self.verticalLayout.addWidget(self.moch1)

        self.listView = QListView(self.verticalLayoutWidget)
        self.listView.setObjectName("listView")

        self.verticalLayout.addWidget(self.listView)

        self.userlineEdit = QLineEdit(self.centralwidget)
        self.userlineEdit.setObjectName("userlineEdit")
        self.userlineEdit.setGeometry(QRect(30, 450, 771, 101))
        font1 = QFont()
        font1.setPointSize(14)
        self.userlineEdit.setFont(font1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 844, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAbout = QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.menuFile.addAction(self.actionSave_as) 
        

        self.setMinimumSize(QSize(844,600))



#pass in sys.argv to allow command line arguments for application
app = QApplication(sys.argv)

#create a Qt widget, which will be our window (in Qt all top level widgets are windows)
window = MainWindow()
window.show()

app.exec()
