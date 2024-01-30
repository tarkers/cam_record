from src.Widgets.player.Ui_player import Ui_Form
from PyQt5 import QtWidgets
from typing import Callable, Union
import numpy as np

## custom
from src.Original import PlayBar, Canvas


class Player(QtWidgets.QWidget, Ui_Form):
    """
    Member
    ------

    Signal
    --------
    play_signal: 按下播放時會發出的信號
    pause_signal: 按下暫停時會發出的信號
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
        play_signal: Callable[[None], None],
        pause_signal: Callable[[None], None],
        finish_signal: Callable[[None], None],
        update_signal: Callable[[int, bool], int],
        clicked_signal: Callable[[list], None] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.can_control: bool = True
        self.frame_count: int = 0
        self.play_signal = play_signal
        self.pause_signal = pause_signal
        self.update_signal = update_signal
        self.finish_signal = finish_signal
        self.clicked_signal = clicked_signal

        self.add_control_widget()
        self.init_value()

    def add_control_widget(self):
        """初始控制UI"""
        # add canvas
        self.canvas_control = Canvas(parent=self, clicked_signal=self.clicked_signal)
        self.canvas.layout().addWidget(self.canvas_control)

        # add playbar
        self.play_control = PlayBar(
            play_signal=self.play_signal,
            pause_signal=self.pause_signal,
            update_signal=self.update_signal,
            finish_signal=self.finish_signal,
            parent=self,
        )
        self.play_bar.layout().addWidget(self.play_control)

    def init_value(
        self,
        image: np.ndarray = None,
        frame_count: int = 0,
    ):
        """初始資料\n
        Param
        --------
        image: 展示圖片
        frame_count:總frame數
        """
        self.frame_count = frame_count
        self.canvas_control.init_value(image)
        self.play_control.init_value(self.frame_count)

    def show_image(self, image: np.ndarray):
        """顯示圖片於canvas_control畫面上"""
        self.canvas_control.show_image(image)

    def update_slider(self, new_idx=None):
        self.play_control.update_slider(new_idx)
        pass

    def control_slider(self, can_control: bool):
        """是否可拉選slider
        Param:
        ------
        can_control (bool):是否可拉選slider
        """
        self.can_control = can_control
        self.play_control.control_slider(can_control)
        if not can_control:
            self.play_control.start_playing()
        pass

    def control_play_btn(self, can_control: bool):
        """是否可點選播放以及暫停\n
        Param:
        ------
        can_control (bool):否可點選播放以及暫停
        """
        self.play_control.control_play_btn(can_control)
        pass

    def show_play_bar(self):
        """顯示playbar"""
        self.play_control.show()

    def hide_play_bar(self):
        """隱藏playbar"""
        self.play_control.hide()

    @property
    def fps(self):
        return self.play_control.fps

    @fps.setter
    def fps(self, v):
        self.play_control.fps = v

    @property
    def aspect_ratio(self):
        return self.canvas_control.aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, v):
        self.canvas_control.aspect_ratio = v

    @property
    def view_type(self):
        return self.canvas_control.view_type

    @view_type.setter
    def view_type(self, v):
        self.canvas_control.view_type = v
