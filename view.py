from Ui_view import Ui_MainWindow
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
import glob
import sys
import cv2
import mimetypes
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QStyleFactory
from src.stream import RealsenseThread, VideoThread
from datetime import datetime
import os
import pathlib

##custom
from util.utils import update_config, video_to_frame, load_json, load_3d_angles
from src import MessageBox, Loading, Clock, Canvas
import yaml

CFG=update_config(r"libs\configs\configs.yaml").CHART




class Ui(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui, self).__init__(parent)
        self.setupUi(self)

        ## initialize model ##
        
        ## loading dialog ##
        self.loading = Loading(self)

        ### video scene ###
        self.scene_2D = QGraphicsScene()
        self.scene_3D = QGraphicsScene()
        ## init lineCanvas ##
        self.line_canvas = Canvas(self.linechart, CFG)
        self.init_video_setting()
        self.messagebox = MessageBox(self)
        self._init_bind()

        ### loading thread
        self.clock = Clock()
        self.clock.fire_signal.connect(self.auto_change_slider)
        self.clock.start()

    def init_subject(self):
        self.subject_box.clear()

    def init_videos_setting(self):
        # clear video
        self.video_box.clear()

    def init_video_setting(self):
        self.video_2D = []
        self.video_3D = []
        self.scene_2D.clear()
        self.scene_3D.clear()
        self.image_index = -1
        self.person_id = 0
        self.start_frame = 0

        # play setting
        self.is_play = False

        ##init_bar
        self.line_canvas.clear_chart()

        self.line_canvas.setGeometry(
            -150, 0, self.linechart.width() + 300, self.linechart.height() - 5
        )

        # init slider bar
        self.frame_slider.setMaximum(100)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"0/0")
        self.video_2d_text.setText("")
        self.video_3d_text.setText("")
        
        # init check box
        self.init_checked_line()

    def _init_bind(self):
        ### load subjects###
        self.load_btn.clicked.connect(self.load_subjects)
        self.subject_box.currentTextChanged.connect(self.prepare_videos)
        self.video_box.currentTextChanged.connect(self.prepare_video)
        self.play_button.clicked.connect(self.play_clicked)
        self.frame_slider.valueChanged.connect(self.slider_changed)


    def slider_changed(self):
        if len(self.video_2D) > 0:
            self.image_index = self.frame_slider.value()
            self.play_frame()

    def play_clicked(self):
        if self.is_play:  # pause play
            self.is_play = False
            self.play_button.setText("Play")
            pass
        else:  # start play the frames
            if self.image_index > self.frame_slider.maximum():
                self.image_index = 0
                self.change_slider_v(self.image_index)
            self.is_play = True
            self.play_button.setText("Pause")
            pass

    def load_box(self, box, root_folder, datatype=None):
        """
        combo box for UI
        """
        box.clear()  # init subject
        box.addItem("clear")
        print(root_folder)
        for file in os.listdir(root_folder):
            d = os.path.join(root_folder, file)
            if os.path.isdir(d):  # add subject name into combobox
                box.addItem(d.split(f"\\")[-1])
            elif os.path.isfile(d):
                if not datatype:
                    print(d, datatype)
                    box.addItem(d)
                elif mimetypes.guess_type(d)[0].startswith(datatype):
                    box.addItem(d)

    def load_subjects(self):
        """
        限制Load Data資料夾底下的Subjects
        """
        self.init_subject()
        self.load_box(self.subject_box, CFG["DIR_VIDEO"], "folder")

    def prepare_videos(self):
        """
        constraint folder path for each data
        1. 2D : Data/{subject}/{video_id}/subject_2D/*
        2. 3D : Data/{subject}/{video_id}/subject_3D/*
        3. FIG : Data/{subject}/FIG/*
        """

        self.init_videos_setting()

        # if self.subject_box.count() > 0:  # 代表是使用者按下subjects選擇按鈕
        #     print("代表是使用者按下subjects選擇按鈕")
        #     self.init_videos_setting()

        # prepare videos
        sn = self.subject_box.currentText()
        idx = self.subject_box.currentIndex()
        if sn == "clear" or idx == -1:  # clear data
            # self.init_var()
            return
        
        
        # load index json
        self.video_box.addItems(["clear"] + load_json(os.path.join("Data", sn)))

    def show_image(self, image, scene, view, aspect_ratio=None):
        scene.clear()
        image = image.copy()
        ih, iw, _ = image.shape
        h, w = view.size().height() - 5, view.size().width() - 5

        if aspect_ratio == "h":
            w, h = h * iw // ih, h
        elif aspect_ratio == "w":
            w, h = w, w * ih // iw
        else:
            w, h = view.size().width() - 5, view.size().height() - 5

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        bytesPerline = 3 * w

        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        scene.addPixmap(QPixmap.fromImage(qImg))
        view.setScene(scene)

    def prepare_video(self):
        ## init chart  ###
        self.loading.start_loading(self.centralwidget.geometry())
        qApp.processEvents()

        self.init_video_setting()
        # if len(self.video_2D) > 0:  # 代表是使用者按下video影片選擇按鈕
        #     print("代表是使用者按下video影片選擇按鈕")
        #     self.init_video_setting()

        vn = self.video_box.currentText()
        idx = self.video_box.currentIndex()  # means not have data
        if vn == "clear" or idx == -1:  # clear data
            self.loading.stop_loading()
            return

        sn = self.subject_box.currentText()
        # show video names
        self.video_2d_text.setText(rf"Data/{sn}/{vn}/Calibrate_Video/{vn}.avi")
        self.video_3d_text.setText( rf"Data/{sn}/{vn}/subject_3D/3D_{vn}.mp4")
        
        # prepare  data
        self.video_2D, fps, count2d = video_to_frame(self.video_2d_text.text())
        self.video_3D, _, count3d = video_to_frame(self.video_3d_text.text())

        ## Future Work: cut video 3d image size ,later change ##
        for i in range(len(self.video_3D)):
            self.video_3D[i] = self.video_3D[i][200:900, 200:900, :]

        self.clock.change_msec(1 / fps * 5)

        # setting video bar
        end_frame = max(count2d, count3d) - 1
        self.frame_slider.setMaximum(end_frame - 1)
        self.frame_label.setText(f"0/{end_frame}")

        # get person id and 3D start_frame
        self.person_id, self.start_frame = load_json(os.path.join(f"Data/{sn}"), vn)

        # get angle from data
        self.line_canvas.show_lines(sn, vn, self.start_frame, self.person_id, False)

        self.resize_scene(self.view_2D, self.view_3D)

        # draw skeleton on 2d image
        if self.start_frame == 0:
            img = self.line_canvas.draw_image(self.video_2D[0], 0)
            self.show_image(img, self.scene_2D, self.view_2D, "w")
        else:
            self.show_image(self.video_2D[0], self.scene_2D, self.view_2D, "w")

        if self.start_frame == 0:
            self.show_image(self.video_3D[0], self.scene_3D, self.view_3D, "h")

        self.loading.stop_loading()

    def auto_change_slider(self):
        if self.is_play:
            self.change_slider_v(min(self.image_index + 1, self.frame_slider.maximum()))

    def change_slider_v(self, image_index):
        self.image_index += 1
        self.frame_slider.setValue(image_index)

    def show_angle_text(self, angles):
        """
        show angle to ui
        0:dr, 1:dl, 2:d2r, 3:d2l, 4:vr, 5:vl
        """

        self.d_right_text.setText(angles[0])
        self.d_left_text.setText(angles[1])
        if len(angles) > 2:  # has 2d angle
            self.d_right_text_2d.setText(angles[2])
            self.d_left_text_2d.setText(angles[3])
        if len(angles) > 4:  # has gt
            self.v_right_text.setText(angles[4])
            self.v_left_text.setText(angles[5])

    def show_display_line(self):
        '''
        如果checkbox有打勾就顯示line
        '''
        check_list = []
        check_list.append(self.d_right_box.isChecked())
        check_list.append(self.d_left_box.isChecked())
        check_list.append(self.d_right_box_2d.isChecked())
        check_list.append(self.d_left_box_2d.isChecked())
        check_list.append(self.v_right_box.isChecked())
        check_list.append(self.v_left_box.isChecked())
        self.line_canvas.change_visible(check_list)
        pass

    def init_checked_line(self):
        """
        初始化所有line的checkbox以及angle label
        """
        self.d_right_box.setChecked(True)
        self.d_left_box.setChecked(True)
        self.d_right_box_2d.setChecked(False)
        self.d_left_box_2d.setChecked(False)
        self.v_right_box.setChecked(True)
        self.v_left_box.setChecked(True)

    def play_frame(self):
        """
        顯示一分析圖片
        """
        # show angle and display line
        angles = self.line_canvas.move_bar(self.image_index)  # linechart move bar
        self.show_angle_text(angles)
        self.show_display_line()

        self.frame_label.setText(f"{self.image_index}/{self.frame_slider.maximum()}")

        ## 2D
        if self.image_index < len(self.video_2D):
            if self.start_frame > self.image_index:
                self.show_image(
                    self.video_2D[self.image_index], self.scene_2D, self.view_2D, "w"
                )
            else:
                img = self.line_canvas.draw_image(
                    self.video_2D[self.image_index], self.image_index - self.start_frame
                )
                self.show_image(img, self.scene_2D, self.view_2D, "w")

        ## 3D
        start_3d = self.image_index - self.start_frame
        if start_3d < len(self.video_3D):
            # to check 3d frame is in range
            if start_3d >= 0:
                self.show_image(
                    self.video_3D[start_3d], self.scene_3D, self.view_3D, "h"
                )
            else:
                self.scene_3D.clear()

        if self.image_index == len(self.video_2D):  # finsih video, stop playing
            self.is_play = False
            self.play_button.setText("Play")

    def resize_scene(self, sc1, sc2):
        w1 = sc1.width() - 5
        w2 = sc2.width() - 5

        nw1 = (w1 + w2) * 6 // 10
        sc1.setMinimumWidth(nw1)

    def closeEvent(self, event):
        result = QMessageBox.question(
            self,
            "離開",
            "確定要離開 ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        event.ignore()

        if result == QMessageBox.Yes:
            self.loading.stop_thread()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('motif'))    #['bb10dark', 'bb10bright', 'cleanlooks', 'cde', 'motif', 'plastique', 'Windows', 'Fusion']

    window = Ui()
    window.show()
    sys.exit(app.exec_())
