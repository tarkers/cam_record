from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5 import QtGui, QtCore, QtWidgets, QtTest
app = QApplication([])
label = QLabel("Hello World!")
label.show()

app.exec_()