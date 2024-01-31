from src.Widgets.scroll_adjust.Ui_scroll_adjust import Ui_Form

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from typing import Callable, Union

import numpy as np
import pandas as pd

from src.Original import Adjust
from util.utils import POSE2D


QVERTICALSTYLE = """
    QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal
    {
        background: none;
    }

    QScrollBar:vertical
    {
        background-color: white;
        width: 15px;
        margin: 15px 3px 15px 3px;
        border: 1px transparent #2A2929;
        border-radius: 4px;
    }

    QScrollBar::handle:vertical
    {
        background-color: #2A2929;         /* #605F5F; */
        min-height: 5px;
        border-radius: 4px;
    }

    QScrollBar::sub-line:vertical
    {
        margin: 3px 0px 3px 0px;
        border-image: url(:/qss_icons/rc/up_arrow_disabled.png);
        height: 10px;
        width: 10px;
        subcontrol-position: top;
        subcontrol-origin: margin;
    }

    QScrollBar::add-line:vertical
    {
        margin: 3px 0px 3px 0px;
        border-image: url(:/qss_icons/rc/down_arrow_disabled.png);
        height: 10px;
        width: 10px;
        subcontrol-position: bottom;
        subcontrol-origin: margin;
    }  
            """


class ScrollAdjust(QtWidgets.QWidget, Ui_Form):
    """
    Member
    ------

    Signal
    --------
    ID_changed_signal:更新ID
        @param
        old_id (int):舊ID
        new_id (int): 新ID
    change_keypoint_signal:更新關鍵點的位置
        @param
        kpt_name (str):關鍵點名稱
        [x,y,[z]] (list): 改變的DATA, is_2d為False的話就沒有z的資訊
        now_focus (QWidgits): 改變的combobox
    """

    def __init__(
        self,
        ID_changed_signal: Callable[[int, int], None] = None,
        change_keypoint_signal: Callable[[int, np.ndarray], None] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)

        self.group_btn = QButtonGroup(self)
        self.z_label.hide()
        self.ID_changed_signal = ID_changed_signal
        self.change_keypoint_signal = change_keypoint_signal
        self.id_loc_map: dict[int, int] = {}  # id所在dataframe row位置
        self.now_iloc: int = -1
        self.frame_df: pd.DataFrame = None
        self.now_kpt_name: str = ""
        self.child_is_connect: bool = True
        self.init_value()
        self.init_bind()
        self.add_control_widget()

    def init_value(self):
        """初始資料"""
        pass

    def init_bind(self):
        """初始綁定"""
        self.changeID_btn.clicked.connect(self.change_id_clicked)
        self.old_id.currentTextChanged.connect(self.set_keypoints)
        self.group_btn.buttonClicked.connect(self.set_now_kpt_name)

    def set_now_kpt_name(self, btn: QRadioButton):
        """設置現在選擇的關鍵點"""
        self.now_kpt_name = [i for i in POSE2D if POSE2D[i]==btn.text()][0]

    def add_control_widget(self, pose_dict=POSE2D):
        """初始控制UI\n
        Param
        --------
        pose_dict:關鍵點{英文關鍵點:中文關鍵點}
        """
        scroll = QScrollArea(self.kpt_widget)
        scroll.setStyleSheet(QVERTICALSTYLE)
        self.kpt_widget.layout().addWidget(scroll)
        scroll.setWidgetResizable(True)
        self.scrollContent = QWidget(scroll)
        self.scrollLayout = QVBoxLayout(self.scrollContent)
        self.scrollContent.setLayout(self.scrollLayout)
        for idx, key in enumerate(pose_dict):
            widgit = Adjust(
                kpt_cn_name=pose_dict[key],
                kpt_name=key,
                kpt_id=idx,
                is_2d=True,
                parent=self,
                change_keypoint_signal=self.change_keypoint_signal,
            )
            self.group_btn.addButton(widgit.kpt_btn, idx)
            self.scrollLayout.addWidget(widgit)
        scroll.setWidget(self.scrollContent)
        self.kpt_widget.layout().addStretch()
        self.scrollContent.setEnabled(False)

    def change_id_clicked(self):
        """更新物體ID"""
        old_text = self.old_id.currentText()
        new_value = self.new_id.value()
        if len(old_text) == 0:
            return
        if int(old_text) == new_value:  # same that will not change
            return

        if self.ID_changed_signal:
            if new_value not in self.id_loc_map:
                now_index = self.old_id.currentIndex()
                self.old_id.removeItem(now_index)
                self.old_id.insertItem(now_index, str(new_value))
                self.id_loc_map[new_value] = self.id_loc_map[int(old_text)]
                del self.id_loc_map[int(old_text)]
            else:
                tmp = self.id_loc_map[new_value]
                self.id_loc_map[new_value] = self.id_loc_map[int(old_text)]
                self.id_loc_map[int(old_text)] = tmp
            self.ID_changed_signal(int(old_text), new_value)

    def set_keypoints(self):
        """設置關鍵點資訊"""
        ## create first id
        text = self.old_id.currentText()

        if len(text) > 0:  # 當有資訊的時候
            self.kpt_label.setText(f"關鍵點ID: {text} ")
            if self.child_is_connect:
                self.disconnect_adjust()
            self.now_iloc = self.id_loc_map[int(text)]  # 這行在dataframe的位置
            keypint_row = self.frame_df[self.frame_df[("ID","ID")] == int(text)]
            if keypint_row is not None:
                data = keypint_row.values[0, 6:].reshape(-1, 3)
                self.loop_update(data)
            self.connect_adjust()

    def loop_update(self, data: np.ndarray):
        """更新所有關鍵點資訊\n
        Param
        --------
        data:關鍵點位置 shape(-1,3)
        """
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).widget()
            widget.set_point(data[i, :2])

    def get_frame_ids(self, frame_df: pd.DataFrame):
        """取得此frame下所有的ID"""
        self.frame_df = frame_df
        ids = self.frame_df[("ID","ID")].values
        iloc_ids = list(self.frame_df.index)
        self.id_loc_map = {ids[i]: iloc_ids[i] for i in range(len(ids))}
        self.set_org_ids(ids)

    def set_org_ids(self, id_list: list):
        """設置原始ID\n
        Param
        --------
        id_list (list):所有ID
        """
        id_list = id_list.astype(str)
        self.old_id.clear()
        self.old_id.addItems(id_list)

    def disconnect_adjust(self):
        """
        Disconnect all binding when playing
        """
        if not self.child_is_connect:  # 如果本身就沒有connect就返回
            return
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).widget()
            widget.disconnect_bind()
        self.child_is_connect = False

    def connect_adjust(self):
        """
        Connect all link when pause editing
        """
        if self.child_is_connect:  # 如果本身有connect就不能在連
            return
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).widget()
            widget.init_bind()
        self.child_is_connect = True

    def set_pixel_range(
        self, xrange: tuple, y_range: tuple, z_range: Union[tuple, None] = None
    ):
        """
        設置所有Qcombobox的範圍\n
        Param
        --------
        x_range: x的範圍
        y_range: y的範圍
        z_range: z的範圍
        """
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).widget()
            widget.set_range(xrange, y_range, z_range)
