import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time
import numpy as np
class Clock(QThread):
    fire_signal = pyqtSignal()   
    _run_flag=True
    msec=1
    def change_msec(self,msec=1):
        self.msec=msec
        
    def run(self):
        # capture from web cam 
        self._run_flag=True
        while self._run_flag:
            self.fire_signal.emit()
            time.sleep(self.msec)
         
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.quit()
        self.wait()
        print("stop video thread")
    
    def isFinished(self):
        print("isFinished video thread")
        
        
    