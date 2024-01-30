import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time
import numpy as np
import queue


## custom
from util.enumType import ERRORTYPE


class RealsenseThread(QThread):
    change_pixmap_signal = pyqtSignal(list)
    error_signal = pyqtSignal(ERRORTYPE)
    finish_signal = pyqtSignal()
    pipeline = None
    _run_flag = True
    w = 1280
    h = 720

    def init_data(self, w, h,use_depth=False):
        self.w = w
        self.h = h
        self.use_depth=use_depth

    def run(self):
        import pyrealsense2 as rs
        print("start running realsense")
        
        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            pipeline = rs.pipeline()
            # Configure streams
            config = rs.config()
            config.enable_stream(rs.stream.color, self.w, self.h, rs.format.rgb8, 30)
            if self.use_depth:
                config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, 30)
            # Start streaming
            pipeline.start(config)
            self._run_flag = True
            while self._run_flag:
                ret, frames = pipeline.try_wait_for_frames()
                if not ret:
                    print("finish")
                    break
                else:
                    depth = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()

                    # Convert images to numpy arrays
                    
                    color_image = np.asanyarray(color_frame.get_data())
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    if self.use_depth:
                        depth_image = np.asanyarray(depth.get_data())
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                        )
                        self.change_pixmap_signal.emit([color_image, depth_colormap])
                    else:
                        self.change_pixmap_signal.emit([color_image])
        except Exception as e:
            self.error_signal.emit(ERRORTYPE.CAMERAERROR)
            print(e)
        self.finish_signal.emit()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        print("realsense camera call stop")
        self._run_flag = False
        if self.pipeline:
            self.pipeline.stop()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(list)
    error_signal = pyqtSignal(ERRORTYPE)
    finish_signal = pyqtSignal()
    _run_flag = True
    cap = None

    def init_data(self, w, h):
        self.w = w
        self.h = h
    
    def run(self):
        # capture from web cam
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS,30) # to prevent high fps
        self._run_flag = True
        while self._run_flag:
            try:
                ret, cv_img = self.cap.read()
                if not ret:
                    break
                else:
                    self.change_pixmap_signal.emit([cv_img])
            except Exception as e:
                self.error_signal.emit(ERRORTYPE.CAMERAERROR)
                break
        self.finish_signal.emit()
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        if self.cap != None:
            self.cap.release()
        print("stop video thread")

    def isFinished(self):
        print("isFinished video thread")


class SavingThread(QThread):
    save_video_finish = pyqtSignal(str)
    _run_flag = True

    def init_data(
        self,
        frames=None,
        image_shape=(1280, 720),  # (w,h)
        fourcc_type="MPV4",
        fps=30,
        save_path="./test.mp4",
        save_type=1,
    ):
        self.q = queue.Queue()
        self.frames = frames
        self.fourcc_type = fourcc_type
        self.save_path = save_path
        self.image_shape = image_shape
        self.fps = fps
        self.type = save_type  # 0: frames錄製   1:立即錄製
        self.video_writer = cv2.VideoWriter(
            self.save_path,
            cv2.VideoWriter_fourcc(*f"{self.fourcc_type}"),
            self.fps,
            (image_shape[0], image_shape[1]),
        )

    def set_new_data(self, image):
        ## make image match size
        # image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_AREA)
        self.q.put(image)

    def run(self):
        print("影片開始存檔")
        if self.type == 1:
            while self._run_flag:
                if not self.q.empty():
                    self.video_writer.write(self.q.get())
        else:
            for frame in self.frames:
                self.video_writer.write(frame)
        self.video_writer.release()
        self.save_video_finish.emit(self.save_path)
        print("Video is saved!")

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        print("stop video saving!!")


class LoadingThread(QThread):
    fire_signal = pyqtSignal()
    _run_flag = True

    def run(self):
        # capture from web cam

        self._run_flag = True
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



# if __name__ == "__main__":
#     test = RealsenseThread()
#     test.start()
