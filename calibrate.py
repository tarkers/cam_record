from PyQt5.QtWidgets import QApplication, QMainWindow
import cv2
import sys
import os
import glob
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from Ui_calibrate import Ui_MainWindow

from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem
dir_path = os.path.dirname(os.path.abspath(__file__))


class Windows(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Windows, self).__init__()
        self.setupUi(self)
        self.camera_matrix = None
        self.dist = None
        self.board_images = []
        self.test_image = None
        self.bind_component()

    def bind_component(self):
        self.load_folder.clicked.connect(self.load_image_folder)
        self.calibration_btn.clicked.connect(
            lambda: self.start_calibration(8, 11, 15))
        self.img_btn.clicked.connect(self.load_image)
        self.mtx_btn.clicked.connect(self.load_mtx)
        self.coef_btn.clicked.connect(self.load_coef)
        self.result_btn.clicked.connect(self.show_result)
        self.save_btn.clicked.connect(self.save_camera_mtx)

    def show_result(self):
        fixed_width=self.graphicsView.width()
        self.test_image = cv2.imread(self.test_image)
        h,  w = self.test_image.shape[:2]
    
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (w,h), 1, (w,h))
        # undistort
        undistort_image = cv2.undistort(self.test_image , self.camera_matrix, self.dist, None, newcameramtx)
        height, width, channel = undistort_image.shape
        
       
        img_aspect_ratio =  float(height / width)
        ratio_height=int(fixed_width*img_aspect_ratio)
        image = cv2.resize(undistort_image, (fixed_width, ratio_height), interpolation=cv2.INTER_AREA)
        
        bytesPerline = 3 * fixed_width
        self.qImg = QImage(image.data, fixed_width, ratio_height,
                           bytesPerline, QImage.Format_RGB888).rgbSwapped()

        scene = QGraphicsScene(self)
        pixmap = QPixmap(QPixmap.fromImage(self.qImg))
        
        
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def load_mtx(self):
        filename, filetype = QFileDialog.getOpenFileName(
            None, "select file", dir_path,"All files (*_cam_matrix.npy)")
        if filename:
            self.mtx_file_label.setText(os.path.basename(filename))
            self.camera_matrix = np.load(filename)

    def load_coef(self):
        filename, filetype = QFileDialog.getOpenFileName(
            None, "select file", dir_path,"All files (*_cam_distortion.npy)")
        if filename:
            self.coe_file_label.setText(os.path.basename(filename))
            self.dist = np.load(filename)

    def load_image(self):
        filename, filetype = QFileDialog.getOpenFileName(
            None, "select file", dir_path,"Image files (*.bmp;*.ico;*.gif;*.jpeg;*.jpg;*.png;*.tif;)")
        if filename:
            self.img_label.setText(os.path.basename(filename))
            self.test_image = filename

    def resize_image(self, image, resize_param):
        (h, w) = image.shape[:2]
        shape = (np.array([w, h])/resize_param).astype(int)
        return tuple(shape)

    def load_image_folder(self):
        folder_name = QFileDialog.getExistingDirectory(
            None, "select folder", dir_path)
        types = ("*.bmp", "*.ico", "*.gif", "*.jpeg", "*.jpg", "*.png","*.tif")
        if folder_name:
            self.folder_label.setText(os.path.basename(folder_name))
            for image_type in types:
                self.board_images.extend(
                    glob.glob('{}/{}'.format(folder_name, image_type)))

    def start_calibration(self, col=8, row=11, mm=15):
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
        for fname in self.board_images:
            img = cv2.imread(fname)
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
        self.mtx_label.setText(str(self.camera_matrix))
        self.dist_label.setText(str(self.dist))

    def save_camera_mtx(self):
        camera_name = self.camera_name.text()
        np.save(f'{camera_name}_cam_matrix', self.camera_matrix)
        np.save(f'{camera_name}_cam_distortion', self.dist)


def main():
    app = QApplication(sys.argv)
    win = Windows()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
