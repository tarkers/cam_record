from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import glob
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from PyQt5.QtWidgets import QMainWindow
from Ui_stream import Ui_MainWindow
from stream import RealsenseThread, VideoThread
from datetime import datetime
import os
import pathlib
##custom
from util.utils import DataType ,video_to_frame

#set 1280 720

class Ui(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui, self).__init__(parent)
        self.setupUi(self)

        ### record ###
        self.scene = QGraphicsScene()
        self.save_label.setText("")
        self.save_place=""
        self.thread = None
        self.open_camera=True
        self.is_record = False
        self.show_alignment = False
        self.writer=None
        self.images = []
        self.test=[]

        ### calibration ###
        self.board_images=[]
        self.camera_matrix = None
        self.dist = None

        self._init_bind()
        self.set_camera()



    def start_thread(self):
        if self.thread == None:
            print("start")
            self.thread = RealsenseThread()
            self.thread.init_data(1280,720)
            self.thread.change_pixmap_signal.connect(self.update_realsense_image)
            self.thread.start()

    def stop_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.stop()
            self.thread=None
            self.images=[]
            self.scene.clear()
            self.graphicsView.setScene(self.scene)

    def _init_bind(self):
        ### record ###
        self.start.clicked.connect(self.start_record)
        self.stop.clicked.connect(self.stop_record)
        self.square_btn.clicked.connect(self.change_show_alignments)
        self.camera_btn.clicked.connect(self.set_camera)

        ### calibration ###
        self.clip_btn.clicked.connect(self.load_image_folder)
        self.video_btn.clicked.connect(self.load_video)
        self.calibration_btn.clicked.connect(self.start_calibration)
        self.mtx_btn.clicked.connect(self.load_mtx)
        self.result_btn.clicked.connect(self.show_results)
        self.save_btn.clicked.connect(self.save_data)
    
    def set_camera(self):
        self.open_camera=not self.open_camera
        if self.open_camera:
            self.camera_btn.setText("Close Camera")
            self.start_thread()
        else:
            self.camera_btn.setText("Open Camera")
            self.stop_thread()
        
        self.error_label.setText("")

    def load_data(self, label_item, dir_path="", value_filter=None, mode=DataType.DEFAULT):
        data_path = None
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(
                self, mode.value['tips'], dir_path)
        else:
            name_filter = mode.value['filter'] if value_filter == None else value_filter
            data_path, _ = QFileDialog.getOpenFileName(
                None, mode.value['tips'], dir_path, name_filter)
        if label_item == None:
            return data_path
        # check exist
        if data_path:
            label_item.setText(os.path.basename(data_path))
            label_item.setToolTip(data_path)
        else:
            label_item.setText(mode.value['tips'])
            label_item.setToolTip("")
        return data_path
    
    def load_video(self):
        '''
        extract video to images and clips several board images
        '''
        if self.thread:
            self.error_label.setText("pls close camera first")
            return
        self.video_path = self.load_data(None, None, None, DataType.VIDEO)       
        # no video found
        if self.video_path == "":
            return
        else:
            self.board_label.setText("Loading!!!")  
            self.images,fps,_=video_to_frame(self.video_path)
            self.extract_images_to_board(fps)

    def extract_images_to_board(self,fps=30):
        '''
        clip images to take them to calibate ,give maximum number of 40
        '''
        self.board_images=[]
        for i in range(0,len(self.images),int(fps//4)):
            self.board_images.append(self.images[i])          
            if len(self.board_images)>40:
                break
        if len(self.board_images)>10:
            self.board_label.setText("Loaded!")       
        else:
            self.board_label.setText("Not enough pictures!")   
        self.convert_cv_qt(self.board_images[0])


    def load_image_folder(self):
        if self.thread:
            self.error_label.setText("please close camera first")
            return
        folder_name = QFileDialog.getExistingDirectory(
            None, "select folder", None)
        types = ("*.bmp", "*.ico", "*.gif", "*.jpeg", "*.jpg", "*.png","*.tif")
        if folder_name:
            self.camera_name.setText(os.path.basename(folder_name).split('.')[0])
            # self.folder_label.setText(os.path.basename(folder_name))
            for image_type in types:
                images=glob.glob('{}/{}'.format(folder_name, image_type))
                for img_name in images:
                    image=cv2.imread(img_name)
                    self.board_images.append(image)
                
        if len(self.board_images)>10:
            self.board_label.setText("Loaded!")    
            self.images=self.board_images.copy()  
            self.convert_cv_qt(self.board_images[0]) 
        else:
            self.board_label.setText("Not enough pictures!")   
        

    def load_mtx(self):
        filename, filetype = QFileDialog.getOpenFileName(
            None, "select file", None,"All files (*matrics.npy)")
        if filename:
            # self.mtx_file_label.setText(os.path.basename(filename))
            self.camera_matrix, self.dist = np.load(filename,allow_pickle=True)


    def start_calibration(self):
        
        col=int(self.col_box.value() )if self.col_box.value() !=0 else 8
        row=self.row_box.value() if self.row_box.value() !=0 else 11
        mm=self.mm_box.value() if self.col_box.value() !=0 else 10
        print("start cal",col,row,mm)
        if len(self.board_images) == 0:
            print("please load images")
            return

        self.objp = np.zeros((col*row, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:row*mm:mm, 0:col*mm:mm].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # retrive images from Test
        for img in self.board_images:
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (row, col), None)
            # print(corners)
            # return 
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(self.objp)

                #  more precise
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                corners = np.array(corners).astype(int)
        #  camera matrix, distortion coefficients, rotation and translation vectors
        _ret, self.camera_matrix, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        self.save_data()

        

    def change_show_alignments(self):
        self.show_alignment = not self.show_alignment

    def start_record(self):
        now = datetime.now()
        video_str = now.strftime("%Y%m%d%H%M%S")+".mp4"
        self.save_place = os.path.join("out_video", video_str)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(self.save_place, fourcc, 30, (1280, 720))
        self.save_label.setText("")  
        self.is_record = True

    def stop_record(self):
        self.is_record = False
        if self.writer:
            self.writer.release()
            self.writer=None
            self.save_label.setText("video_save to: " + self.save_place)
        else:
            self.save_label.setText("no recording found")

    def convert_cv_qt(self, cv_img,fix_h=500):
        """Convert from an opencv image to QPixmap"""
        image=cv_img.copy()
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.is_record:
            # self.images.append(cv_img)
            self.writer.write(cv_img)
            image = self.add_record_text(image)
            
        if self.show_alignment:
            image = self.camera_alignment(image)
        fix_h=self.graphicsView.size().height()-5
        h, w, ch = image.shape
        w,h=int(w*fix_h/h),fix_h

        image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(image, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(convert_to_Qt_format))
        self.graphicsView.setScene(self.scene)

        # return scene



        

    def update_realsense_image(self, color_depth):
        """Updates the image_label with a new opencv image"""
        color,depth=color_depth
        h=self.graphicsView.size().height()-5
        scene = self.convert_cv_qt(color,h)
        # self.graphicsView.setScene(scene)

    def closeEvent(self, event):
        self.stop_thread()
        event.accept()

    def add_record_text(self, image):
        h, w, ch = image.shape
        image = cv2.putText(image, "Recording....", (w-250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        return image

    def camera_alignment(self, image, square_w=30):
        h, w, ch = image.shape
        for i in range(0, w, square_w):
            image = cv2.line(image, (i, 0),  (i, h), (0, 255, 0), 1)
        for y in range(0, h, square_w):
            image = cv2.line(image, (0, y), (w, y), (0, 255, 0), 1)
        return image

    
    def show_results(self):
        if len(self.images)>0:
            h,w=self.images[0].shape[:2]
            for image in self.images:
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (w,h), 0, (w,h))
                # undistort
                undistort_image = cv2.undistort(image, self.camera_matrix, self.dist, None, newcameramtx)
                cv2.imshow("image",undistort_image)
                cv2.waitKey(30)
        

    def save_data(self):
        '''
        save matrics to folder
        '''
        camera_name = self.camera_name.text()
        p = pathlib.Path(f"{camera_name}/")
        p.mkdir(parents=True, exist_ok=True)
        np.save(f'{camera_name}/matrics', np.array([self.camera_matrix,self.dist],dtype=object))
        #save board images
        for i,image in enumerate(self.board_images):
            cv2.imwrite(f"./{camera_name}/{i}.png",image)
            
        self.error_label.setText("data saved!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
