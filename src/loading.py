import sys, time
from PyQt5 import QtCore, QtWidgets,QtTest
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from src.stream import LoadingThread
alive, Progress = True, 0
class Loading(QWidget):
    def __init__(self,parent=None):
        QWidget.__init__(self,parent)
        self.thread=None
        self.is_show=False
        self.index=len("Loading....")
        self.text="Loading...."
        self.init_ui()
        self.hide()
        # self.start_thread()

    def init_ui(self):
        #init label
        self.title = QLabel("Loading...", self)
        self.title.setFont(QFont('Arial', 30))
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("color: #FFFFFF")

        #init qwidget
        pal=self.palette()
        color=QColor(0,0,0,80)
        pal.setColor(QPalette.Background, color)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def start_thread(self):
        if self.thread == None:
            print("start")
            self.thread = LoadingThread()
            self.thread.fire_signal.connect(self.change_text)
            self.thread.start()

    def stop_thread(self):
        if self.thread:
            self.thread.stop()
           
            

    def start_loading(self,g,text="Loading!"):
        '''
        start loading...
        '''
        self.setGeometry(g)
        self.is_show=True
        self.title.setText(text)
        qApp.processEvents()
        self.title.move(self.geometry().width()//2-self.title.width(),\
                        min(500,self.geometry().height()//3))
        
        self.show()
        self.raise_()
        # self.start_thread()

    def change_text(self):
        if self.is_show:
            print("test")
            self.title.setText(self.text[:self.index])
            self.index+=1
            if self.index>len(self.text)-1:
                self.index=0
            
            
    def stop_loading(self):
        self.is_show=False
        self.hide()
        # self.stop_thread()
        