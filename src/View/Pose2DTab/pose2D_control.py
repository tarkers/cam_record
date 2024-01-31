from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import cv2
from datetime import datetime
import os
import glob
import json
import numpy as np
from typing import Callable, Union
from easydict import EasyDict
import pandas as pd
import pathlib

## custom
from src.View.Pose2DTab.Ui_pose2D import Ui_Form
from util.enumType import *
from src.Widgets import Player, ScrollAdjust

from util import YOLODetectorQueue, Pose2DQueue, VitPoseQueue
from libs.vis import plot_single_2d_skeleton, plot_2d_skeleton

from util.utils import set_video_capture, create_2D_csv, save_json, video_to_frame
from tracker.mc_bot_sort import BoTSORT


class Pose2D_Control(QtWidgets.QWidget, Ui_Form):
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
    finish_signal: 播放結束時會發出的信號
    clicked_signal: 點選圖片發出(x,y)位置
        @param
        [x,y] (list): 點選x,y的位置
    update_signal: 更新新位置
        @param
        now_count (int): slider的位置
        is_next_frame (bool): 是否是接續的
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
        self.fps: int = fps
        self.is_read_all: bool = False
        self.tracker: Union[BoTSORT, None] = None
        self.df: Union[pd.DataFrame, None] = None

        ##init widget
        self.player_control = None
        self.adjust_control = None

        ##init config
        self.cfg = cfg
        self.detector_cfg = EasyDict({**cfg.Detector, **cfg.COMMON})
        self.pose2D_cfg = EasyDict({**cfg.Pose2D, **cfg.COMMON})
        self.tracker_cfg = EasyDict({**cfg.TRACKER, **cfg.COMMON})
        self.ROOT_DIR = self.cfg.COMMON.ROOT_DIR
        ## parent function
        self.append_log = append_log
        self.set_loading = set_loading

        self.init_value()
        self.init_bind()

        ## timer for images
        self.timer = QTimer()

        self.add_control_widget()
        # self.init_model()

        ## test code
        # self.adjust_ids(8,0)

    def init_value(self):
        """初始資料"""
        self.now_tab: str = self.func_tab.currentWidget().objectName()
        self.video_set.hide()
        self.stream_set.hide()
        self.ana_group.hide()
        self.analyze_btn.setEnabled(True)
        self.cap: cv2.VideoCapture = None
        self.contatin_first: bool = False
        self.is_read_all: bool = False
        self.media_type = MEDIATYPE.Picture
        self.init_type()

    def init_type(self):
        self.pose_data = []
        self.media_files = []
        self.frames = []
        self.media_path: Union[str, None] = None
        self.is_analyze: bool = False
        if self.cap:
            self.cap.release()
        self.cap = None
        self.custom_pause: bool = False
        self.now_count = 0
        self.analyze_btn.setEnabled(True)

        pass

    def init_bind(self):
        """初始綁定"""
        self.input_type_group.buttonClicked.connect(self.change_type)
        self.func_tab.currentChanged.connect(self.change_tab)
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_pic_btn.clicked.connect(self.load_picture)
        self.load_data_btn.clicked.connect(self.load_data)
        self.load_pic_dir_btn.clicked.connect(lambda: self.load_picture(True))
        self.save_btn.clicked.connect(self.save_data)
        self.save_modify_btn.clicked.connect(self.save_modify)

    def add_control_widget(self):
        """初始控制UI"""
        ## create playing widget
        self.player_control = Player(
            play_signal=self.play,
            pause_signal=self.pause,
            update_signal=self.update_frame,
            finish_signal=self.finish,
            clicked_signal=self.adjust_kpts_by_click,
        )
        self.view_tab.layout().addWidget(self.player_control)

        ## create adjust model
        self.adjust_control = ScrollAdjust(
            parent=self,
            ID_changed_signal=self.adjust_ids,
            change_keypoint_signal=self.adjust_kpts,
        )
        self.adjust_box.layout().addWidget(self.adjust_control)

    def adjust_kpts(self, kpt_name, val, focus_widget):
        """更新dataframe關鍵點位置
        @param
        kpt_name (str):關鍵點名稱
        [x,y,[z]] (list): 改變的DATA
        focus_widget (QWidgits): 強制focus此widget
        """
        now_iloc = self.adjust_control.now_iloc
        self.df.iloc[now_iloc]
        self.df.at[now_iloc, (kpt_name, "X")] = val[0]
        self.df.at[now_iloc, (kpt_name, "Y")] = val[1]
        self.player_control.show_image(self.draw_now_frame(self.now_count - 1))
        focus_widget.lineEdit().setFocus()

    def init_model(self):
        """初始model"""
        # detector of  YoloV7
        self.detector = YOLODetectorQueue(cfg=self.detector_cfg)
        # self.pose2d = Pose2DQueue(self.pose2D_cfg)
        self.pose2d = VitPoseQueue(self.pose2D_cfg)
        self.pose2d.load_model()

    def adjust_kpts_by_click(self, val):
        """更新dataframe關鍵點位置
        @param
        [x,y,[z]] (list): 改變的DATA
        """
        if not self.custom_pause:
            return
        if self.now_tab != "adjust":
            return

        now_iloc = self.adjust_control.now_iloc
        kpt_name = self.adjust_control.now_kpt_name
        self.df.at[now_iloc, (kpt_name, "X")] = val[0]
        self.df.at[now_iloc, (kpt_name, "Y")] = val[1]
        self.player_control.show_image(self.draw_now_frame(self.now_count - 1))

    def finish(self):
        """播放完畢"""
        if not self.player_control.can_control:
            if self.media_type == MEDIATYPE.Video:
                self.cap.release()
                self.cap = None
            self.player_control.control_slider(True)
            self.player_control.fps = self.fps
            self.analyze_btn.setEnabled(False)
            self.append_log("分析完畢....", "red")
            self.save_btn.setEnabled(True)

    def play(self):
        """播放"""
        self.custom_pause = False
        if self.now_tab == "adjust":
            self.adjust_control.disconnect_adjust()
            self.adjust_control.scrollContent.setEnabled(False)
            pass
        elif self.is_analyze:
            self.save_btn.setEnabled(False)
            if self.media_type == MEDIATYPE.Video:
                self.timer.singleShot(10, self.analyze_video)

    def pause(self):
        """暫停"""
        self.custom_pause = True
        if self.now_tab == "adjust":
            self.adjust_control.connect_adjust()
            self.adjust_control.scrollContent.setEnabled(True)

        elif self.is_analyze:
            self.save_btn.setEnabled(True)

    def update_frame(self, now_count: int, is_next_frame: bool = False):
        """更新畫面\n
        Param
        ---------
        now_count (int): slider的位置
        is_next_frame (bool): 是否是接續的
        """
        if len(self.frames) == 0:
            return
        if len(self.frames) < now_count:
            return

        if self.now_tab == "adjust":
            self.now_count = now_count
            frame = self.draw_now_frame(self.now_count - 1)
            self.player_control.show_image(frame)
        else:
            frame = self.frames[now_count - 1]
            self.player_control.show_image(frame)
            ## update next frame
            if self.is_analyze:
                self.now_count = now_count + 1
                if self.media_type == MEDIATYPE.Video:
                    self.timer.singleShot(10, self.analyze_video)
                else:
                    self.timer.singleShot(10, self.analyze_pics)

    def change_tab(self):
        """tab change"""
        text = self.func_tab.currentWidget().objectName()
        self.now_tab = text
        if text == "analyze":
            self.init_value()
            ## set image as default
            self.type_group_check(self.input_type_group.checkedButton().text())
            print("分析")
        elif text == "adjust":
            self.init_value()
            print("調整")
        self.player_control.setEnabled(False)

    def reset_control(self):
        """重設player_control為無法互動"""
        self.player_control.control_slider(False)
        self.player_control.play_control.stop_playing()

    def load_picture(self, is_dir: bool = False):
        """載入圖片或圖片資料夾\n
        Param
        ------
        is_dir:(bool) 是否載入資料夾
        """
        if is_dir:
            self.media_path = self.load_media(mode=DataType.FOLDER)
        else:
            self.media_path = self.load_media(mode=DataType.IMAGE)
        if self.media_path is not None:
            self.now_count = 0
            self.media_files = (
                glob.glob(rf"{self.media_path}/*") if is_dir else [self.media_path]
            )

            self.append_log(f"已載入{ '資料夾' if is_dir else '圖片'}: {self.media_path}")
            self.player_control.init_value(frame_count=len(self.media_files))
            frame = cv2.imread(self.media_files[0])
            self.frames.append(frame)
            self.player_control.show_image(frame)
            self.player_control.setEnabled(False)

    def load_video(self):
        """載入影片"""
        self.init_type()
        self.reset_control()
        self.media_path = self.load_media(DataType.VIDEO)
        self.set_loading(True)
        if self.media_path is not None:
            self.append_log(f"已載入影片: {self.media_path}")
            self.create_video_cap(self.media_path)
            ## get first frame
            self.now_count = 0
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            ret, frame = self.cap.read()
            if ret:
                self.player_control.show_image(frame)
                self.frames.append(frame)
                self.player_control.setEnabled(False)

            else:
                QMessageBox.critical(
                    self,
                    "錯誤",
                    "影片無開啟!",
                    QMessageBox.Yes,
                )
                self.ana_group.hide()
        self.set_loading(False)

    def load_media(self, mode=DataType.DEFAULT, dir_path=None, value_filter=None):
        """載入媒體"""
        data_path = None
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(
                self, mode.value["tips"], dir_path
            )
        else:
            name_filter = mode.value["filter"] if value_filter == None else value_filter
            data_path, _ = QFileDialog.getOpenFileName(
                None, mode.value["tips"], dir_path, name_filter
            )
        if data_path is None or data_path == "":
            self.ana_group.hide()
            return None
        else:
            self.ana_group.show()
            self.frames = []
            return data_path

    def load_all(self):
        """一次讀取成圖片進來"""
        self.append_log("讀取中....")
        self.set_loading(True)
        self.is_read_all = True
        self.set_loading(False)
        pass

    def change_type(self):
        """改變media類型"""
        self.init_type()
        self.ana_group.hide()
        self.type_group_check(self.input_type_group.checkedButton().text())

    def type_group_check(self, input_type):
        """
        group box check
        """
        if input_type == "影片":
            self.video_set.show()
            self.pic_set.hide()
            self.stream_set.hide()
            self.media_type = MEDIATYPE.Video
        elif input_type == "圖片":
            self.video_set.hide()
            self.pic_set.show()
            self.stream_set.hide()
            self.media_type = MEDIATYPE.Picture
        elif input_type == "串流":
            self.stream_set.show()
            self.video_set.hide()
            self.pic_set.hide()
            self.media_type = MEDIATYPE.Stream

    def create_video_cap(self, video_path):
        """建立videocap_object\n
        Param
        --------
        video_path:影片路徑
        """
        if self.cap:
            self.cap.release()
        self.cap, fps, count, fourcc, (w, h) = set_video_capture(video_path)
        video_name = os.path.basename(video_path)
        videoinfo = {
            "totalframe": count,
            "fourcc": fourcc,
            "fps": fps,
            "frameSize": (w, h),
            "videoName": video_name,
        }
        basename = self.get_data_basename()
        foldername = os.path.join(self.ROOT_DIR, "Pose", basename)
        p = pathlib.Path(foldername)
        p.mkdir(parents=True, exist_ok=True)
        save_json(videoinfo, os.path.join(foldername, rf"{basename}_videoinfo.json"))
        self.copy_video(foldername)
        self.player_control.init_value(frame_count=count)

    def main_analyze_part(
        self, frame: np.ndarray, is_first: bool = False, tracking: bool = False
    ):
        """模型偵測\n
        Param
        --------
        frame: 圖片
        is_first:是否為第一張圖
        tracking:是否加入追蹤

        """
        has_detect = True
        has_pose = True
        item, detect_fps = self.detector.detect_for_one_frame(
            frame
        )  # detect for human bbox
        if item is None or item[0] is None:
            has_detect = False

        if has_detect:
            (orig_img, result), pose_fps = self.pose2d.pose_estimation(
                item
            )  # detect for human pose 2d
            if result is None:
                has_pose = False

        if not has_detect or not has_pose:
            if is_first:
                self.frames[0] = frame
            else:
                self.frames.append(frame)

        elif has_pose:
            self.format_pose_data(result, self.now_count - 1)
            if is_first:
                self.frames[0] = plot_2d_skeleton(
                    orig_img,
                    result,
                    showbox=True if self.box_check.isChecked() else False,
                    tracking=tracking,
                )
                self.player_control.update_slider(1)
            else:
                self.frames.append(
                    plot_2d_skeleton(
                        orig_img,
                        result,
                        showbox=True if self.box_check.isChecked() else False,
                        tracking=tracking,
                    )
                )
                self.player_control.update_slider()

        # print("detectFps",detect_fps, "Posefps",pose_fps)

    def analyze_video(self):
        """分析影片"""
        if not self.custom_pause and self.cap:
            # analyze first frame
            if self.contatin_first:
                self.now_count = 1
                self.contatin_first = False
                self.main_analyze_part(self.frames[0], True, True)
            else:
                ret, frame = self.cap.read()
                if ret:
                    self.main_analyze_part(frame, tracking=True)
                    cv2.waitKey(1)
                else:
                    print("no frame")

    def analyze_pics(self):
        """分析圖片"""
        if self.contatin_first:
            self.now_count = 1
            self.contatin_first = False
            self.main_analyze_part(self.frames[0], True)
        elif self.now_count <= len(self.media_files):
            frame = cv2.imread(self.media_files[self.now_count - 1])
            self.main_analyze_part(frame)
        else:
            print("no pics")

    def format_pose_data(self, result: dict, img_id: int = 0):
        """整理資料集\n
        Param
        ------
        result:資料
        img_id:圖片ID
        """
        if self.media_type == MEDIATYPE.Picture:
            img_id = os.path.basename(self.media_files[img_id])  # 圖片集給檔案名
        result = result["result"]
        row_data = []
        for item in result:
            xywh = list(np.round(np.array(item["box"]), 0))
            if self.media_type == MEDIATYPE.Video:
                idx = (
                    [int(item["idx"][0])]
                    if isinstance(item["idx"], list)
                    else [int(item["idx"])]
                )
            else:
                idx = [0]
            kp = np.round(item["keypoints"].numpy(), 0)
            kp_score = item["kp_score"].numpy()
            keypoints = list(np.append(kp, kp_score, axis=1).flatten())
            row_data = np.array([img_id] + idx + xywh + keypoints)
            self.pose_data.append(row_data)

    def start_analyze(self):
        """開始偵測2D Pose"""
        ## create data Frame
        self.set_loading(True)
        self.save_btn.setEnabled(False)
        self.player_control.control_slider(False)
        self.player_control.setEnabled(True)
        if self.media_type == MEDIATYPE.Video:
            self.detector.tracking = True
            self.tracker = BoTSORT(self.tracker_cfg, frame_rate=self.fps)
            self.detector.set_tracker(self.tracker)
        else:
            self.detector.tracking = False
            self.detector.set_tracker(None)
        self.set_loading(False)
        self.append_log("開始偵測2D Pose...", "red")
        self.pose_data = []
        self.timer = QTimer()
        self.contatin_first = True
        self.is_analyze = True
        self.analyze_btn.setEnabled(False)
        # self.player_control.control_play_btn(True)
        if self.media_type == MEDIATYPE.Video:
            self.analyze_video()
        else:
            self.analyze_pics()

    def save_data(self):
        """存入原始檔案"""
        self.set_loading(True)
        self.pose_data = np.array(self.pose_data)

        pose_df = create_2D_csv(
            self.pose_data, is_video=(self.media_type == MEDIATYPE.Video)
        )
        basename = self.get_data_basename()
        ## create parent folder
        folder = os.path.join(self.ROOT_DIR, "Pose", basename)
        p = pathlib.Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        save_pose_path = os.path.join(folder, f"{basename}_2D.csv")
        pose_df.to_csv(save_pose_path)
        self.set_loading(False)
        self.append_log(rf"已將POSE 2D存檔:{save_pose_path}")
        self.save_btn.setEnabled(False)

    def get_data_basename(self):
        """取得Basename"""
        basename = os.path.basename(self.media_path)
        basename = basename.rsplit(".", 1)[0]
        return basename

    def copy_video(self, folder: str):
        """影片複製至新資料夾\n
        Param
        ---------
        folder (str): 新資料夾
        """
        destination = os.path.join(folder, os.path.basename(self.media_path))
        if not pathlib.Path(destination).is_file():
            import shutil

            shutil.copy(self.media_path, folder)

    def load_data(self):
        """載入csv資料"""
        data_path = QFileDialog.getExistingDirectory(self, "", None)
        if data_path is None or data_path == "":
            return
        self.data_folder = data_path
        base_folder = os.path.basename(self.data_folder)
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
        self.append_log(rf"Pose 2D已載入: {path_2D_csv}")
        self.df = pd.read_csv(path_2D_csv, header=[0, 1], index_col=0)
        info_path = os.path.join(self.data_folder, rf"{base_folder}_videoinfo.json")
        if os.path.exists(info_path):
            self.video_info = self.load_video_info(info_path)
            self.frames, self.fps, count = video_to_frame(
                rf"{self.data_folder}\{self.video_info['videoName']}"
            )
            h, w, _ = self.frames[0].shape
            self.adjust_control.set_pixel_range((0, w), (0, h))
            self.player_control.init_value(frame_count=count)
            self.player_control.setEnabled(True)
            self.player_control.show_image(self.frames[0])
            self.player_control.control_slider(True)
        else:
            return

    def load_video_info(self, path: str):
        """
        獲取影片資料\n
        Param
        ----------
        path: 路徑
        """
        f = open(path)
        data = json.load(f)
        return data

    def save_modify(self):
        """更新資料存檔"""
        if self.df is not None:
            base_folder = os.path.basename(self.data_folder)
            path = rf"{self.data_folder}\{base_folder}_2DM.csv"
            self.df.to_csv(path)
            self.append_log(rf"POSE 2D修正已存檔: {path}")

    def draw_now_frame(self, frame_id: int):
        """圖片顯示skeleton\n
        Param
        ---------
        frame_id:圖片index

        """
        frame_df = self.df.loc[self.df["ImageID"]["ID"] == frame_id]
        self.adjust_control.get_frame_ids(frame_df)
        frame_data = frame_df.values
        frame = self.frames[frame_id].copy()
        for item in frame_data:
            kpt = item[6:].reshape(-1, 3)
            frame = plot_single_2d_skeleton(
                frame,
                kp_preds=kpt[:, :2],
                kp_scores=kpt[:, 2],
                unique_id=item[1],
                bbox=item[2:6],
                showbox=True if self.box_check.isChecked() else False,
                tracking=True,
            )
        return frame



    def adjust_ids(self, old_id: int, new_id: int):
        """更新物件ID"""
        # # self.now_count = 2
        # # old_id = 8
        # # new_id = 0
        # # self.df = pd.read_csv(rf"Pose\test\test_2D.csv", header=[0, 1], index_col=0)
        # self.total_frame = 8
        self.total_frame=self.df[('ImageID','ID')].max()+1
        df = self.df.loc[(self.df[("ImageID", "ID")] >= self.now_count - 1)]
        for img_id in range(self.now_count - 1, self.total_frame):
            switch_df = df[
                (df[("ImageID", "ID")] == img_id)
                & df[("ID", "ID")].isin([old_id, new_id])
            ]
            if len(switch_df.index) > 2:
                print("檔案有錯誤!!")
                exit()
            elif len(switch_df.index) == 2:  # id switch
                first_id = self.df.at[switch_df.index[0], ("ID", "ID")]
                self.df.at[switch_df.index[0], ("ID", "ID")] = self.df.at[
                    switch_df.index[1], ("ID", "ID")
                ]
                self.df.at[switch_df.index[1], ("ID", "ID")] = first_id
                # print(first_id, "ID SWITCH")
            elif (
                (df[("ImageID", "ID")] == img_id) & (df[("ID", "ID")] == old_id)
            ).any():  # give new id
                row = df[
                    (df[("ImageID", "ID")] == img_id) & (df[("ID", "ID")] == old_id)
                ]
                self.df.at[row.index[0], ("ID", "ID")] = new_id
                # print(img_id,row.index[0], "REASSIGN")
        self.player_control.show_image(self.draw_now_frame(self.now_count-1))
