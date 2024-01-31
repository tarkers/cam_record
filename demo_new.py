from Ui_demo_new import Ui_MainWindow
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QStyleFactory, QApplication, QMessageBox, qApp
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
import time
from datetime import datetime
import os
import cv2
import pathlib
import sys
import numpy as np
import yaml

## custom

from src import Loading
from util.enumType import *
from src.View import Camera_Control
from src.View import Pose2D_Control
from src.View import Pose3D_Control
from util import update_config


class Ui(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui, self).__init__(parent)
        self.setupUi(self)
        ##init config
        self.cfg = update_config(r"libs\configs\configs_new.yaml")
        self.ROOT_DIR = self.cfg.COMMON.ROOT_DIR
        self.tab = "camera_tab"
        ## create save folder
        self.create_folder("Stream")
        self.create_folder("Pose")

        ## loading dialog ##
        self.loading = Loading(self)

        self.add_control_widget()
        self.signal_bind()

    def signal_bind(self):
        self.op_tab.currentChanged.connect(self.change_tab)
        self.clear_message_btn.clicked.connect(lambda: self.message_box.clear())

    def add_control_widget(self):
        """初始控制UI"""
        ## camera tab control
        self.cam_widget = Camera_Control(
            append_log=self.append_log,
            set_loading=self.set_loading,
            parent=self,
            cfg=self.cfg,
        )
        self.camera_tab.layout().addWidget(self.cam_widget)

        ## 2D tab
        self.pose2d_widget = Pose2D_Control(
            append_log=self.append_log,
            set_loading=self.set_loading,
            parent=self,
            cfg=self.cfg,
        )
        self.pose2d_tab.layout().addWidget(self.pose2d_widget)

        ## 3D tab
        self.pose3d_widget = Pose3D_Control(
            append_log=self.append_log,
            set_loading=self.set_loading,
            parent=self,
            cfg=self.cfg,
        )
        self.pose3d_tab.layout().addWidget(self.pose3d_widget)

    def change_tab(self):
        # self.set_loading(True)

        if self.tab == "camera_tab":
            self.cam_widget.clear_all()
            self.cam_widget.init_value()
        elif self.tab == "pose2d_tab":
            self.pose2d_widget.clear_all()
            self.pose2d_widget.init_value()
            pass
        elif self.tab == "pose3d_tab":
            self.pose3d_widget.clear_all()
            self.pose3d_widget.init_value()
            pass
        self.tab = self.op_tab.currentWidget().objectName()

        # self.set_loading(False)
        pass

    def append_log(
        self, html_text="測試", color="#000000", is_bold=False, change_line=True
    ):
        """
        Property
        --------
        顯示操作的訊息
        """
        cursor = self.message_box.textCursor()
        text = f'<span style=" font-weight:{400 if is_bold else 150}; color:{color};">{html_text}</span>'
        html = f'<p style="margin:0px 5px 0px 0px; -qt-block-indent:0; text-indent:0px; ">{text}</p>'

        cursor.insertHtml(html)
        if change_line:  # 換行
            cursor.insertHtml("<br/> ")

        # 移到末尾
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        scrollBar = self.message_box.verticalScrollBar()
        scrollBar.setValue(scrollBar.maximum() - 15)

    def set_loading(self, is_start=False, text="Loading"):
        if is_start:
            self.loading.start_loading(self.centralwidget.geometry(), text)
            qApp.processEvents()
        else:
            self.loading.stop_loading()

    def create_folder(self, folder_path):
        pathlib.Path(os.path.join(self.ROOT_DIR, folder_path)).mkdir(
            parents=True, exist_ok=True
        )

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

            self.cam_widget.clear_all()
            event.accept()


if __name__ == "__main__":
    # app.create()
    app = QApplication(sys.argv)
    app.setStyle(
        QStyleFactory.create("motif")
    )  # ['bb10dark', 'bb10bright', 'cleanlooks', 'cde', 'motif', 'plastique', 'Windows', 'Fusion']

    window = Ui()
    window.show()
    sys.exit(app.exec_())
