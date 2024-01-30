from src.Original.adjust.Ui_adjust import Ui_Form

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from typing import Callable, Union
import numpy as np


class Adjust(QtWidgets.QWidget, Ui_Form):
    """
    Member
    ------
    kpt_id: 關鍵點ID
    kpt_cn_name: 關鍵點中文名稱
    kpt_name: 關鍵點名稱
    is_2d:是否是2D資料\n

    Signal
    --------
    change_keypoint_signal:更新關鍵點的位置
        @param
        kpt_name (str):關鍵點名稱
        [x,y,[z]] (list): 改變的DATA, is_2d為False的話就沒有z的資訊
        now_focus (QWidgits): 改變的combobox
    """

    def __init__(
        self,
        kpt_id: int = -1,
        kpt_cn_name: str = "",
        kpt_name: str = "",
        is_2d: bool = True,
        parent: QtWidgets = None,
        change_keypoint_signal: Callable[[int, np.ndarray, QWidget], None] = None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.kpt_id = kpt_id
        self.kpt_cn_name = kpt_cn_name
        self.kpt_name = kpt_name
        self.is_2d = is_2d

        self.change_keypoint_signal = change_keypoint_signal
        self.init_value()
        self.init_bind()

    def kpt_changed(self, now_focus: QComboBox):
        """偵測到改變關鍵點位置\n
        now_focus:改變關鍵點的combobox\n
        Emit
        -------
        change_keypoint_signal:  [kpt_name, [x,y,[z]], now_focus]
        """
        if self.change_keypoint_signal:
            if self.is_2d:
                arr = np.array([self.kpt_x.value(), self.kpt_y.value()])
            else:
                arr = np.array(
                    [self.kpt_x.value(), self.kpt_y.value(), self.kpt_z.value()]
                )
            self.change_keypoint_signal(self.kpt_name, arr, now_focus)

    def init_bind(self):
        """初始綁定\n
        可換成要enter才觸發的event
        editingFinished
        """
        self.kpt_x.valueChanged.connect(lambda: self.kpt_changed(self.kpt_x))
        self.kpt_y.valueChanged.connect(lambda: self.kpt_changed(self.kpt_y))
        self.kpt_z.valueChanged.connect(lambda: self.kpt_changed(self.kpt_z))
        pass

    def disconnect_bind(self):
        """解除綁定"""
        self.kpt_x.valueChanged.disconnect()
        self.kpt_y.valueChanged.disconnect()
        self.kpt_z.valueChanged.disconnect()

    def init_value(self):
        """初始化要調整的keypoints"""
        if self.is_2d:
            self.kpt_z.hide()
            self.kpt_x.setMaximumWidth(100)
            self.kpt_y.setMaximumWidth(100)

        ## set range
        self.set_range()

        self.kpt_btn.setText(self.kpt_cn_name)

    def set_point(self, points: list):  # x,y,z
        """設置關鍵點位置\n
        Param
        --------
        points:關鍵點位置 (x,y,z)
        """
        self.kpt_x.setValue(points[0])
        self.kpt_y.setValue(points[1])
        if not self.is_2d:
            self.kpt_z.setValue(points[2])

    def set_range(
        self,
        x_range: tuple = (0, 1920),
        y_range: tuple = (0, 1080),
        z_range: tuple = (0, 5),
    ):
        """設置關鍵點的設置範圍\n
        Param
        --------
        x_range: x的範圍
        y_range: y的範圍
        z_range: z的範圍
        """
        self.kpt_x.setMinimum(x_range[0])
        self.kpt_x.setMaximum(x_range[1])
        self.kpt_y.setMinimum(y_range[0])
        self.kpt_y.setMaximum(y_range[1])
        
        if not self.is_2d:
            self.kpt_z.setMinimum(z_range[0])
            self.kpt_z.setMaximum(z_range[1])
