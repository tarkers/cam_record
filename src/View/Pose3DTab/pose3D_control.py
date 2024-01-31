from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, qApp

from easydict import EasyDict
from datetime import datetime
import os
import glob
import pandas as pd
import re
import json
import numpy as np
from typing import Callable

## custom
from src.View.Pose3DTab.vis_3D_canvas import Canvas3DWrapper
from src.View.Pose3DTab.Ui_pose3D import Ui_Form
from util.utils import update_config, create_3D_csv, video_to_frame
from libs.vis import plot_single_2d_skeleton
from util.enumType import *
from util.pose3d_generator import Pose3DGenerator
from src.Original import PlayBar, Canvas


class Pose3D_Control(QtWidgets.QWidget, Ui_Form):
    def __init__(
        self,
        append_log: Callable[[str, str, bool, bool], None],
        set_loading: Callable[[bool], None],
        cfg,
        fps=30,
        play_frame=0,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.fps = fps
        self.play_frame = play_frame
        self.person_2D_data = {}
        self.timer = QTimer()
        self.init_value()
        self.init_bind()

        ##init config
        self.cfg = cfg
        self.pose3D_cfg = EasyDict({**cfg.Pose3D, **cfg.COMMON})
        self.ROOT_DIR = self.cfg.COMMON.ROOT_DIR
        self.clip_len = self.pose3D_cfg.clip_len

        ## parent function
        self.append_log = append_log
        self.set_loading = set_loading
        self.total_frame = 0

        ## create playing widget
        self.play_control = PlayBar(
            play_signal=self.play,
            pause_signal=self.pause,
            update_signal=self.update_data,
            finish_signal=self.finish,
            parent=self,
        )
        self.play_widget.layout().addWidget(self.play_control)

        ## create 2D scene
        self.canvas2D = Canvas(parent=self)
        self.scene2D.layout().addWidget(self.canvas2D)
        # # init 3DWidget
        self.canvas3D = Canvas3DWrapper()

        self.scene3D.layout().addWidget(self.canvas3D.canvas.native)
        self.view_stack.setCurrentWidget(self.scene3D)
        self.grip_size = QtWidgets.QSizeGrip(self.view_frame)
        self.view_frame.layout().addWidget(
            self.grip_size, 0, QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight
        )
        self.grip_size.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.grip_size.setMaximumSize(self.grip_size.sizeHint())
        # self.view_stack.setCurrentWidget(self.scene2D)
        # teset load
        self.init_model()
        # exit()

    def init_value(self):
        self.func_tab.hide()
        self.person_2d_df = None
        self.frame_id = 0
        self.select_id = 0
        self.kpts = None
        self.data_folder = None
        self.df = None
        self.total_frame = 0
        self.id_box.clear()

        pass

    def init_model(self):
        self.poseModel = Pose3DGenerator(self.pose3D_cfg)
        pass

    def play(self):
        print("start")

        pass

    def pause(self):
        print("暫停")
        pass

    def update_data(self, frame_count, is_next_frame):
        self.update_view(frame_count)
        pass

    def finish(self):
        pass

    def set_3D_layout(self):
        self.sceneRight_C = Canvas3DWrapper()

        pass

    def init_bind(self):
        self.id_box.currentTextChanged.connect(self.change_id)
        self.func_tab.currentChanged.connect(self.change_tab)
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.load_data_btn.clicked.connect(self.load_data)
        self.save_btn.clicked.connect(self.save_pose_data)
        self.view_group.buttonClicked.connect(self.change_view)
        pass

    def change_tab(self):
        text = self.func_tab.currentWidget().objectName()

        if text == "analyze":
            self.can_control_bar(False)
            print("分析")
        elif text == "adjust":
            self.init_adjust_tab()
        pass

    def init_adjust_tab(self):
        if self.data_folder:
            if self.load_person_3d_ids(os.path.basename(self.data_folder)):
                self.init_viewing()
                self.can_control_bar(True)

    def change_view(self):
        tabtext = self.view_group.checkedButton().text()
        if tabtext == "2D":
            self.view_stack.setCurrentWidget(self.scene2D)
            if self.scene2D.layout().itemAt(0) is None:
                self.scene2D.layout().addWidget(self.canvas2D)
                self.sceneLeft.layout().removeWidget(self.canvas2D)
        elif tabtext == "3D":
            self.view_stack.setCurrentWidget(self.scene3D)
            if self.scene3D.layout().itemAt(0) is None:
                self.scene3D.layout().addWidget(self.canvas3D.canvas.native)
                self.sceneRight.layout().removeWidget(self.canvas3D.canvas.native)

        elif tabtext == "Both":
            self.view_stack.setCurrentWidget(self.sceneBoth)
            if self.sceneRight.layout().itemAt(0) is None:
                self.sceneRight.layout().addWidget(self.canvas3D.canvas.native)
                self.scene3D.layout().removeWidget(self.canvas3D.canvas.native)
                self.sceneLeft.layout().addWidget(self.canvas2D)
                self.scene2D.layout().removeWidget(self.canvas2D)

    def can_control_bar(self, can_control):
        self.play_control.can_control = can_control
        self.play_control.setEnabled(can_control)

    def load_data(self):
        self.func_tab.hide()
        self.can_control_bar(False)
        self.save_btn.setEnabled(False)
        data_path = QFileDialog.getExistingDirectory(self, "", None)
        if data_path is None:
            self.data_folder = None
        else:
            self.data_folder = data_path
            base_folder = os.path.basename(self.data_folder)
            # if not self.load_person_3d_ids(base_folder):
            #     print("錯誤1")
            #     return
            if not self.load_2d_video_frame(base_folder):
                self.set_loading(False)
                print("錯誤2")
                return
            if not self.load_2d_csv(base_folder):
                self.set_loading(False)
                print("錯誤3")
                return
            self.func_tab.show()
            ## check if its on adjust tab
            if self.func_tab.currentWidget().objectName() == "adjust":
                self.init_adjust_tab()

    def load_video_info(self, path):
        f = open(path)
        data = json.load(f)
        return data

    def prepare_ids(self, csv_path):
        """載入檔案內所有ID"""
        self.df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        str_list=self.df["ID"]["ID"].astype(str)
        id_lists = sorted(set(list(str_list)))
        self.total_frame = int(self.df.iloc[-1]["ImageID"]["ID"] + 1)
        self.id_box.clear()
        self.id_box.addItems(id_lists)
        
        return True

    def change_id(self, id):
        if id == "":  # no select
            return
        self.set_loading(True)
        self.select_id = int(id)
        self.person_2D_data = {0: []}
        self.person_2d_df = self.df.loc[self.df["ID"]["ID"] == self.select_id]
        npdata = self.person_2d_df.to_numpy()
        ## resmaple to 3d first
        data = npdata[:, 6:].reshape(-1, 26, 3)
        data3Ds = self.poseModel.resmaple_datas(data)

        ## generate specific id data
        start_idx, end_idx = 0, -1
        if npdata.shape[0] == self.total_frame:
            self.person_2D_data[0] = data3Ds
           
        else:
            for item,data in zip(npdata, data3Ds):
                frame_id = int(item[0])
                key_points = data
                if frame_id - end_idx < 2:
                    for _ in range((frame_id - end_idx)):
                        self.person_2D_data[start_idx].append(key_points)
                    end_idx = frame_id
                else:
                    if len(self.person_2D_data[start_idx]) > 0:
                        self.person_2D_data[start_idx] = np.array(
                            self.person_2D_data[start_idx]
                        )
                        self.person_2D_data[frame_id] = [key_points]
                    end_idx = start_idx = frame_id
        
        
        self.set_loading(False)

    def start_analyze(self):
        self.set_loading(True)
        self.id_box.setEnabled(False)
        person_3D_data = {}
        for k in list(self.person_2D_data.keys()):
            total_len = len(self.person_2D_data[k])
            if total_len==0:
                continue
            person_3D_data[k] = np.array([])
            datas = np.split(
                self.person_2D_data[k],
                list(range(self.clip_len, total_len, self.clip_len)),
            )
            for data in datas:
                data3d = self.poseModel.predict_3D_Pose(data)

                if len(person_3D_data[k]) == 0:
                    person_3D_data[k] = data3d
                else:
                    person_3D_data[k] = np.concatenate(
                        [person_3D_data[k], data3d], axis=1
                    )
            person_3D_data[k] = person_3D_data[k].reshape(-1, 17, 3)
            
            
        self.id_box.setEnabled(True)
        self.set_loading(False)
        self.save_btn.setEnabled(True)
        self.analyze_btn.setEnabled(False)
        self.kpts = np.zeros((self.total_frame, 17, 3))
        for k in list(person_3D_data.keys()):
            # print(k,person_3D_data[k].shape)
            self.kpts[k : k+len(person_3D_data[k]), :, :] = person_3D_data[k]
        self.append_log("POSE已經分析完畢")

        self.init_viewing()

    def init_viewing(self):
        self.play_control.init_value(self.total_frame)
        self.can_control_bar(True)
        self.canvas3D.set_datas(self.kpts)

    def save_pose_data(self):
        self.set_loading(True)
        save_kpt = self.kpts.reshape(-1, 17 * 3)

        # add imageid
        image_id_list = np.arange(self.total_frame).reshape(self.total_frame, 1)
        data = np.concatenate((image_id_list, save_kpt), axis=1)

        pose_df = create_3D_csv(data)
        basefolder = os.path.basename(self.data_folder)
        save_pose_path = os.path.join(
            self.ROOT_DIR, "Pose", basefolder, f"{basefolder}_3D_P{self.select_id}.csv"
        )
        pose_df.to_csv(save_pose_path)
        self.set_loading(False)
        self.append_log(rf"已將POSE 3D存檔:{save_pose_path}")
        self.save_btn.setEnabled(False)
        self.analyze_btn.setEnabled(True)

    def update_view(self, frame_count):
        if frame_count > self.total_frame:
            return
        else:
            text = self.view_group.checkedButton().text()
            frame = self.frames[frame_count - 1]
            if text == "Both" or "2D":
                if self.person_2d_df is not None:
                    row = self.person_2d_df.loc[
                        self.person_2d_df["ImageID"]["ID"] == frame_count - 1
                    ]
                    if row is not None and len(row) > 0:
                        row = row.to_numpy()[0]
                        bbox = row[2:6]
                        keypoint = row[6:].reshape(-1, 3)

                        frame = plot_single_2d_skeleton(
                            frame,
                            kp_preds=keypoint[:, :2],
                            kp_scores=keypoint[:, 2],
                            unique_id=self.select_id,
                            bbox=bbox,
                            showbox=True if self.box_check.isChecked() else False,
                        )
                self.canvas2D.show_image(frame)
            if text == "Both" or "3D":
                self.canvas3D.update_points(frame_count - 1)

    def load_person_3d_ids(self, base_folder):
        self.id_3d_box.clear()
        # check 3d file
        files = glob.glob(os.path.join(self.data_folder, rf"{base_folder}_3D_P*.csv"))
        try:
            ids = [
                re.findall(r"3D_P.+.csv", os.path.basename(file))[0]
                .replace("3D_P", "")
                .replace(".csv", "")
                for file in files
            ]
        except Exception as e:
            QMessageBox.warning(
                self,
                "警告",
                "有檔案不符合格式規定!",
                QMessageBox.Yes,
            )
            self.append_log("3D csv檔案格式須為xxxx_3D_P(ID).csv")
            return False

        if len(ids) > 0:
            self.id_3d_box.addItems(ids)
            self.load_3D_id_csv(base_folder, ids[0])  # default for first id
            return True
        else:
            return False

    def load_2d_video_frame(self, base_folder):
        self.set_loading(True)
        ## check video info
        if not os.path.exists(rf"{self.data_folder}\{base_folder}_videoinfo.json"):
            self.frames = []
            QMessageBox.information(
                self,
                "告知",
                "此檔案不含影片資料!",
                QMessageBox.Yes,
            )
        else:
            self.poseModel.video_info = self.load_video_info(
                rf"{self.data_folder}\{base_folder}_videoinfo.json"
            )
            try:
                videoname = self.poseModel.video_info["videoName"]
                self.frames, self.fps, self.total_frame = video_to_frame(
                    rf"{self.data_folder}\{videoname}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "錯誤",
                    "影片meta json格式錯誤!",
                    QMessageBox.Yes,
                )
                return False
        self.play_control.fps = max(self.fps - 10, 30)  # give more time for render
        self.set_loading(False)
        return True

    def load_3D_id_csv(self, base_folder, pid):
        df3d = pd.read_csv(
            os.path.join(self.data_folder, rf"{base_folder}_3D_P{pid}.csv"),
            header=[0, 1],
            index_col=0,
        )
        self.kpts = df3d.values[:, 1:].reshape(-1, 17, 3)
        self.canvas3D.set_datas(self.kpts)

    def load_2d_csv(self, base_folder):
        path_2D_csv = rf"{self.data_folder}\{base_folder}_2DM.csv"
        if not os.path.exists(path_2D_csv):
            path_2D_csv = rf"{self.data_folder}\{base_folder}_2D.csv"
            if not os.path.exists(path_2D_csv):
                QMessageBox.critical(
                    self,
                    "錯誤",
                    "沒有csv檔案!",
                    QMessageBox.Yes,
                )
                return False
        else:
            self.append_log(rf"載入2D Pose csv: {path_2D_csv}")
            return self.prepare_ids(path_2D_csv)
