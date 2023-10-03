import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time
import numpy as np
class RealsenseThread(QThread):
    change_pixmap_signal = pyqtSignal(list)
    pipeline=None
    _run_flag=True
    def init_data(self,w,h):
        self.w=w 
        self.h=h
    def run(self):
        import pyrealsense2 as rs
        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            self.pipeline = rs.pipeline()

            # Configure streams
            config = rs.config()
            config.enable_stream(rs.stream.color, self.w, self.h, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, 30)
            # Start streaming
            self.pipeline.start(config)
            self._run_flag=True
            while not self._run_flag:
                
                try:
                    if not self._run_flag:
                        break
                    ret,frames = self.pipeline.try_wait_for_frames()
                    if ret:
                        depth = frames.get_depth_frame()
                        color_frame = frames.get_color_frame()
                        # cv2.imshow("test",color_frame)
                        # Convert images to numpy arrays
                        depth_image = np.asanyarray(depth.get_data())
                        color_image = np.asanyarray(color_frame.get_data())
                        color_image=cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                        self.change_pixmap_signal.emit([color_image,depth_colormap])
                        if not depth: continue
                    else:
                        break
                except:
                    print("interrupted thread")
                    break
            print("thread finish")
            self.quit()
            self.wait()
        except Exception as e:
            print(e)
            pass
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        if self.pipeline:
            self.pipeline.stop()
        

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)   
    _run_flag=True
    cap=None
    def run(self):
        # capture from web cam
        self.cap = cv2.VideoCapture(0)
        self._run_flag=True
        while self._run_flag:
            ret, cv_img =  self.cap.read()
            if not ret:
                break
            else:
                self.change_pixmap_signal.emit(cv_img)
                

    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        if self.cap!=None:
            self.cap.release()
        print("stop video thread")
       
        
    def isFinished(self):
        print("isFinished video thread")

class LoadingThread(QThread):
    fire_signal = pyqtSignal()   
    _run_flag=True
    def run(self):
        # capture from web cam
        
        self._run_flag=True
        while self._run_flag:
            self.fire_signal.emit()
            self.msleep(100)
        
        print("stop loading isFinished!!")

    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        print("stop loading thread!!")
        
        
    def isFinished(self):
        print("isFinished loading thread")

if __name__ == "__main__":
    test =RealsenseThread()
    test.start()
