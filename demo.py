from Ui_demo import Ui_MainWindow
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QStyleFactory, QApplication, QMessageBox, qApp

import time
from datetime import datetime
import os
import cv2
import pathlib
import sys
import numpy as np
import yaml

## custom
from util.utils import update_config, video_to_frame, load_json, load_3d_angles
from components import MessageBox, Loading, Clock, Canvas, Canvas3DWrapper
from components.stream import RealsenseThread, SavingThread
from util.enumType import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SAVINGTYPE = "FAST"


CFG = update_config(r"libs\configs\configs.yaml").CHART


class Ui(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui, self).__init__(parent)
        self.setupUi(self)
        ## loading dialog ##
        self.loading = Loading(self)

        self.play_frame = 0
        self.fps = 30
        self.init_camera_tab_value()
        self.camera_tab_bind()
        # self.set_3D_layout()

    def set_3D_layout(self):
        self._canvas_wrapper = Canvas3DWrapper()
        self.scene_layout.addWidget(self._canvas_wrapper.canvas.native)
        pass

    def init_camera_tab_value(self):
        """
        Init the value from camera tab

        """
        self.scene_video.clear()

        self.camera_is_record = False
        self.camera_is_open = False
        self.camera_thread = None
        self.is_viewing_record = False
        self.camera_images = []

        if hasattr(self, "video_writer"):
            if self.video_writer is not None:
                self.video_writer.release()
                # if self.video_save_path is not None:   #remove not load file
                #     pathlib.Path.unlink(self.video_writer)
        self.video_writer = None
        self.video_save_path = None
        # 初始影片類型
        self.video_type.addItems(VIDEOTYPE.list())
        # 初始攝影機
        self.camera_type.addItems(CAMERATYPE.list())
        if self.camera_type.currentText() == CAMERATYPE.REALSENSE.value:
            self.camera_thread = RealsenseThread()
        else:
            assert (
                self.camera_type.currentText() in CAMERATYPE.list()
            ), "We don't have this camera type yet"

        self.record_btn.setText("開始錄影")
        self.record_view_btn.setText("錄製瀏覽")
        self.save_video_btn.setText("影片存檔")
        self.camera_btn.setEnabled(True)
        self.record_btn.setEnabled(False)
        self.record_view_btn.setEnabled(False)
        self.save_video_btn.setEnabled(False)
        self.set_loading(False)

    def camera_tab_bind(self):
        self.camera_btn.clicked.connect(self.set_camera)
        self.save_video_btn.clicked.connect(self.start_save_video)
        self.record_btn.clicked.connect(self.set_record)
        self.record_view_btn.clicked.connect(self.set_record_view)
        self.save_w_record_check.stateChanged.connect(self.set_online_saving)
        pass

    def set_online_saving(self):
        """
        Property
        --------
        是否是線上錄製
        """

        self.save_video_btn.setEnabled(False)
        self.record_view_btn.setEnabled(False)
        self.camera_images = []
        if self.save_w_record_check.isChecked():
            self.save_video_btn.hide()
            if self.video_writer is None:
                self.camera_images = []
        else:
            self.video_writer = None
            self.save_video_btn.show()

    def append_log(self, html_text="測試", color="#000000", is_bold=False, is_end=True):
        """
        Property
        --------
        顯示操作的訊息
        """
        cursor = self.message_box.textCursor()
        text = f'<span style=" font-weight:{400 if is_bold else 150}; color:{color};">{html_text}</span>'
        html = f'<p style="margin:0px 5px 0px 0px; -qt-block-indent:0; text-indent:0px; ">{text}</p>'

        cursor.insertHtml(html)
        if is_end:  # 換行
            cursor.insertHtml("<br/> ")

    def set_loading(self, is_start=False):
        if is_start:
            self.loading.start_loading(self.centralwidget.geometry())
            qApp.processEvents()
        else:
            self.loading.stop_loading()

    def set_camera(self):
        """
        Property
        ---------
        開啟或關閉cam corder
        """
        self.camera_is_open = not self.camera_is_open
        if self.camera_is_open:
            self.camera_btn.setText("關閉攝影機")
            self.append_log("攝影機啟動")
            self.start_camera_thread()

            ## 可以錄影 ##
            self.record_btn.setEnabled(True)
        else:
            self.set_loading(True)
            self.camera_btn.setText("啟動攝影機")
            self.append_log("攝影機關閉")
            self.camera_btn.setEnabled(False)
            self.stop_camera_thread()

    def set_record(self):
        """
        Property
        ---------
        開啟或關閉cam corder
        """
        self.camera_is_record = not self.camera_is_record
        if self.camera_is_record:
            ## start record for image
            self.record_btn.setText("停止錄影")
            self.append_log("錄影已開始.....")
            if self.save_w_record_check.isChecked():
                self.create_saving_thread(1280, 720)
                self.saving_thread.start()
            self.camera_images = []
            self.save_w_record_check.setEnabled(False)
            self.save_video_btn.setEnabled(False)
            self.record_view_btn.setEnabled(False)
            self.camera_btn.setEnabled(False)

        else:
            self.record_btn.setText("開始錄影")
            self.append_log("結束錄影...")
            self.save_w_record_check.setEnabled(True)
            if self.save_w_record_check.isChecked():
                self.saving_thread.stop()
                self.record_view_btn.setEnabled(True)
                self.record_btn.setEnabled(False)

            elif len(self.camera_images) > 0:
                self.save_video_btn.setEnabled(True)
                self.record_view_btn.setEnabled(True)
                self.camera_btn.setEnabled(True)

    def set_record_view(self):
        """
        Property
        ---------
        瀏覽/結束瀏覽錄製後的影片
        """
        self.is_viewing_record = not self.is_viewing_record
        if self.is_viewing_record:
            ## start record for image
            self.record_view_btn.setText("關閉瀏覽")
            self.append_log("瀏覽錄製影片.....")
            if self.save_w_record_check.isChecked():
                self.set_loading(True)
                self.camera_images, _, _ = video_to_frame(self.save_video_path)
                self.set_loading(False)
            self.play_frame = 0
            self.record_btn.setEnabled(False)
        else:
            self.record_view_btn.setText("開啟瀏覽")
            self.append_log("結束瀏覽影片...")

            self.record_btn.setEnabled(True)

    def create_saving_thread(self, w=1280, h=720):
        fourcc_type = (
            "mp4v" if self.video_type.currentText() == VIDEOTYPE.MP4.value else "XVID"
        )
        self.create_folder("Stream")
        self.save_video_path = os.path.join(
            ROOT_DIR,
            "Stream",
            time.strftime(
                f"%Y%m%d%H%M%S_{self.camera_type.currentText()}.{self.video_type.currentText()}"
            ),
        )  # base on timestamp
        print("Video Save:", self.save_video_path)
        self.saving_thread = SavingThread()
        self.saving_thread.init_data(
            self.camera_images.copy(),
            image_shape=(w, h),
            fourcc_type=fourcc_type,
            fps=self.fps,
            save_path=self.save_video_path,
            save_type=1 if self.save_w_record_check.isChecked() else 0,
        )
        if self.save_w_record_check.isChecked():

            def is_finish_save(x):
                self.append_log(f"影片已存於:{x}")
                self.record_btn.setEnabled(True)
                self.camera_images = []

                pass

            self.saving_thread.save_video_finish.connect(lambda x: is_finish_save(x))

        else:
            self.saving_thread.save_video_finish.connect(
                lambda x: self.append_log(f"影片已存於:{x}")
            )

    def start_save_video(self):
        if len(self.camera_images) > 0:
            self.create_saving_thread(
                self.camera_images[0].shape[1], self.camera_images[0].shape[0]
            )
            self.saving_thread.start()
            ## 正在存檔 ，將button設為Unable ##
            self.save_video_btn.setEnabled(False)
        else:
            self.append_log("沒有錄製的影片")

    def start_camera_thread(self):
        self.camera_thread.init_data(1280, 720)
        self.camera_thread.change_pixmap_signal.connect(self.get_camera_image)
        self.camera_thread.error_signal.connect(self.error_handle)
        self.camera_thread.finish_signal.connect(self.init_camera_tab_value)
        self.camera_thread.start()

    def error_handle(self, type: ERRORTYPE):
        if type == ERRORTYPE.CAMERAERROR:
            print(type)
        pass

    def stop_camera_thread(self):
        if self.camera_thread:
            self.camera_thread.quit()
            self.camera_thread.stop()

    def get_camera_image(self, data):
        """
        Property
        ---------
        獲取Camera thread的圖片
        """
        ## 播放錄製的影片 ##
        if self.is_viewing_record:
            self.show_image(self.camera_images[self.play_frame], self.scene_video)
            self.play_frame += 1
            if self.play_frame >= len(self.camera_images):
                self.play_frame = 0
            return

        if self.camera_type.currentText() == CAMERATYPE.REALSENSE.value:
            color, depth = data  # color[h,w,c]

        else:  # for rgb vamera
            color = data

            return

        ## start catch image 暫存or直接存 test##
        if self.camera_is_record:
            if self.save_w_record_check.isChecked():
                self.saving_thread.set_new_data(color)

            else:
                self.camera_images.append(color)  # 可瀏覽錄製影片

        self.show_image(color, self.scene_video)

    def create_folder(self, folder_path):
        pathlib.Path(os.path.join(ROOT_DIR, folder_path)).mkdir(
            parents=True, exist_ok=True
        )

    def setImageText(
        self,
        image,
        text,
        place=(20, 30),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    ):
        image = cv2.putText(
            image, text, place, font, fontScale, color, thickness, lineType
        )

    def show_image(self, image, view, aspect_ratio=None):
        """
        Property
        --------
        顯示圖片於畫面上
        """
        image = image.copy()
        ## 顯示正在錄影 ##
        if self.is_viewing_record:
            self.setImageText(image, "Recorded Video")
        if self.camera_is_record:
            self.setImageText(image, "Recording....")

        ih, iw, _ = image.shape
        if aspect_ratio is None:
            h, w = view.size().height() - 5, view.size().width() - 5

            if aspect_ratio == "h":
                w, h = h * iw // ih, h
            elif aspect_ratio == "w":
                w, h = w, w * ih // iw
            else:
                w, h = view.size().width() - 5, view.size().height() - 5
        else:
            h, w = ih, iw

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        bytesPerline = 3 * w

        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        view.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        result = QMessageBox.question(
            self,
            "離開",
            "確定要離開 ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        event.ignore()

        if result == QMessageBox.Yes:
            self.set_loading(False)

            if self.camera_thread:
                self.camera_thread.quit()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(
        QStyleFactory.create("motif")
    )  # ['bb10dark', 'bb10bright', 'cleanlooks', 'cde', 'motif', 'plastique', 'Windows', 'Fusion']

    window = Ui()
    window.show()
    sys.exit(app.exec_())
