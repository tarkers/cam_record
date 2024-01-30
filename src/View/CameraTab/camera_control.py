from src.View.CameraTab.Ui_Camera import Ui_Form
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QMessageBox

import time
from datetime import datetime
import os
import cv2

import numpy as np
import yaml
from typing import Callable

## custom
from util.utils import update_config, video_to_frame, load_3d_angles
from src.stream import RealsenseThread, SavingThread, VideoThread
from util.enumType import *
from src.Widgets import Player



class Camera_Control(QtWidgets.QWidget, Ui_Form):
    """
    Member
    ------
    fps (int): fps
    cfg: 設定檔案\n
    Signal
    --------
    append_log: 顯示log文字
        @param
        html_text (str): 文字
        color (str): 文字顏色
        is_bold (bool): 是否要用粗體字
        change_line (bool): 是否換行
    set_loading: loading畫面
        @param
        loading (bool):loading畫面
    """

    def __init__(
        self,
        append_log: Callable[[str, str, bool, bool], None],
        set_loading: Callable[[bool], None],
        cfg,
        fps=30,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.fps = fps
        ##init config
        self.cfg = cfg
        self.ROOT_DIR = self.cfg.COMMON.ROOT_DIR
        
        ## parent function
        self.append_log = append_log
        self.set_loading = set_loading

        self.add_control_widget()

        self.init_value()
        self.init_bind()

    def add_control_widget(self):
        """初始控制UI"""
        ## create playing widget
        self.player_control = Player(
            play_signal=self.play,
            pause_signal=self.pause,
            update_signal=self.update_signal,
            finish_signal=None,
            clicked_signal=None,
        )
        self.view_tab.layout().addWidget(self.player_control)

    def play(self):
        print("開始")
        pass

    def pause(self):
        print("暫停")
        pass

    def update_signal(self, now_count, is_next_frame=False):
        """更新畫面\n
        Param
        ---------
        now_count (int): slider的位置
        is_next_frame (bool): 是否是接續的
        """
        if 0 <= now_count and now_count <= len(self.frames):
            self.player_control.show_image(self.frames[now_count - 1])

    def init_value(self):
        """初始設置"""
        # camera read w,h
        self.w, self.h = 1920, 1080
        self.is_record = False
        self.camera_is_open = False
        self.camera_thread = None
        self.is_viewing_record = False
        self.frames = []

        if hasattr(self, "video_writer"):
            if self.video_writer is not None:
                self.video_writer.release()

        self.video_writer = None
        self.video_save_path = None
        # 初始影片類型
        self.video_type.addItems(VIDEOTYPE.list())
        # 初始攝影機類型
        self.camera_type.addItems(CAMERATYPE.list())

        self.record_btn.setText("開始錄影")
        self.record_view_btn.setText("錄製瀏覽")
        self.save_video_btn.setText("影片存檔")
        self.camera_btn.setEnabled(True)
        self.record_btn.setEnabled(False)
        self.record_view_btn.setEnabled(False)
        self.save_video_btn.setEnabled(False)
        self.player_control.init_value()
        self.player_control.control_slider(False)
        self.set_loading(False)

    def init_bind(self):
        """初始綁定"""
        self.camera_btn.clicked.connect(self.set_camera)
        self.save_video_btn.clicked.connect(self.start_save_video)
        self.record_btn.clicked.connect(self.set_record)
        self.record_view_btn.clicked.connect(self.set_record_view)
        self.save_w_record_check.stateChanged.connect(self.set_online_saving)

    def set_online_saving(self):
        """是否是線上錄製"""
        self.save_video_btn.setEnabled(False)
        self.record_view_btn.setEnabled(False)
        self.frames = []
        if self.save_w_record_check.isChecked():
            self.save_video_btn.hide()
            if self.video_writer is None:
                self.frames = []
        else:
            self.video_writer = None
            self.save_video_btn.show()

    def set_camera(self):
        """開啟或關閉cam corder"""
        self.set_loading(True)
        self.camera_is_open = not self.camera_is_open
        if self.camera_is_open:
            self.camera_btn.setText("關閉攝影機")
            self.append_log("攝影機啟動")
            if self.camera_type.currentText() == CAMERATYPE.REALSENSE.value:
                self.camera_thread = RealsenseThread()
            elif self.camera_type.currentText() == CAMERATYPE.AX700.value:
                self.camera_thread = VideoThread()
            else:
                assert (
                    self.camera_type.currentText() in CAMERATYPE.list()
                ), "We don't have this camera type yet"

            self.start_camera_thread()

        else:
            self.set_loading(True)
            self.camera_btn.setText("啟動攝影機")
            self.append_log("攝影機關閉")
            self.camera_btn.setEnabled(False)
            self.stop_camera_thread()
        self.set_loading(False)

    def set_record(self):
        """開啟或關閉錄影"""
        self.is_record = not self.is_record
        if self.is_record:
            ## start record for image
            self.record_btn.setText("停止錄影")
            self.append_log("錄影已開始.....")
            self.player_control.view_type = "is_record"
            if self.save_w_record_check.isChecked():
                self.create_saving_thread(self.w, self.h)
                self.saving_thread.start()
            self.frames = []

            self.save_w_record_check.setEnabled(False)
            self.save_video_btn.setEnabled(False)
            self.record_view_btn.setEnabled(False)
            self.camera_btn.setEnabled(False)

        else:
            self.record_btn.setText("開始錄影")
            self.append_log("結束錄影...")
            self.player_control.view_type = ""
            self.save_w_record_check.setEnabled(True)
            if self.save_w_record_check.isChecked():
                self.saving_thread.stop()
                self.record_view_btn.setEnabled(True)
                self.record_btn.setEnabled(False)

            elif len(self.frames) > 0:
                self.save_video_btn.setEnabled(True)
                self.record_view_btn.setEnabled(True)
                self.camera_btn.setEnabled(True)

    def set_record_view(self):
        """瀏覽/結束瀏覽錄製後的影片"""
        self.is_viewing_record = not self.is_viewing_record
        if self.is_viewing_record:
            ## start record for image
            self.record_view_btn.setText("關閉瀏覽")
            self.append_log("瀏覽錄製影片.....")
            if self.save_w_record_check.isChecked():
                """For shortest Video"""
                if len(self.frames) == 0:
                    self.set_loading(True)
                    self.frames, _, _ = video_to_frame(self.save_video_path)
                    self.set_loading(False)
            self.player_control.init_value(frame_count=len(self.frames))
            self.player_control.control_slider(True)
            self.record_btn.setEnabled(False)
            self.player_control.view_type = "is_view_record"
            self.player_control.show_image(self.frames[0])
        else:
            self.record_view_btn.setText("開啟瀏覽")
            self.append_log("結束瀏覽影片...")
            self.record_btn.setEnabled(True)
            self.player_control.init_value()
            self.player_control.view_type = ""

            self.player_control.control_slider(False)

    def create_saving_thread(
        self,
    ):
        """儲存錄製的影片檔案"""
        fourcc_type = (
            "mp4v" if self.video_type.currentText() == VIDEOTYPE.MP4.value else "XVID"
        )

        self.save_video_path = os.path.join(
            self.ROOT_DIR,
            "Stream",
            time.strftime(
                f"%Y%m%d%H%M%S_{self.camera_type.currentText()}.{self.video_type.currentText()}"
            ),
        )  # base on timestamp
        print("Video Save:", self.save_video_path)
        self.saving_thread = SavingThread()
        self.saving_thread.init_data(
            self.frames.copy(),
            image_shape=(self.w, self.h),
            fourcc_type=fourcc_type,
            fps=self.fps,
            save_path=self.save_video_path,
            save_type=1 if self.save_w_record_check.isChecked() else 0,
        )
        if self.save_w_record_check.isChecked():

            def is_finish_save(x):
                self.append_log(f"影片已存於:{x}")
                self.record_btn.setEnabled(True)
                self.frames = []

                pass

            self.saving_thread.save_video_finish.connect(lambda x: is_finish_save(x))

        else:
            self.saving_thread.save_video_finish.connect(
                lambda x: self.append_log(f"影片已存於:{x}")
            )

    def start_save_video(self):
        """開始儲存"""
        if len(self.frames) > 0:
            self.create_saving_thread(self.frames[0].shape[1], self.frames[0].shape[0])
            self.saving_thread.start()
            ## 正在存檔 ，將button設為Unable ##
            self.save_video_btn.setEnabled(False)
        else:
            self.append_log("沒有錄製的影片")

    def start_camera_thread(self):
        """啟動攝影機"""
        self.camera_thread.init_data(self.w, self.h)
        self.camera_thread.change_pixmap_signal.connect(self.get_camera_image)
        self.camera_thread.error_signal.connect(self.error_handle)
        self.camera_thread.finish_signal.connect(self.init_value)
        self.camera_thread.start()

    def error_handle(self, type: ERRORTYPE):
        """攝影機錯誤訊息"""
        if type == ERRORTYPE.CAMERAERROR:
            QMessageBox.critical(
                self,
                "錯誤",
                "為偵測到攝影機!",
                QMessageBox.Yes,
            )
        pass

    def stop_camera_thread(self):
        if self.camera_thread:
            self.camera_thread.quit()
            self.camera_thread.stop()

    def get_camera_image(self, data: list):
        """獲取Camera thread的圖片
        data:圖片list
        """

        ## 正在播放錄製的影片 ##
        if self.is_viewing_record:
            return
        if not self.record_btn.isEnabled() and not self.is_record:  # 確認可以錄製
            self.record_btn.setEnabled(True)

        if len(data) == 2:
            color, depth = data  # color[h,w,c]
        else:
            color = data[0]

        if self.is_record:
            if self.save_w_record_check.isChecked():
                self.saving_thread.set_new_data(color)

            else:
                self.frames.append(color)  # 可瀏覽錄製影片

            self.player_control.show_image(color)
        else:
            self.player_control.show_image(color)

    def clear_all_thread(self):
        """停止運行的thread"""
        if self.camera_thread:
            self.camera_thread.quit()
