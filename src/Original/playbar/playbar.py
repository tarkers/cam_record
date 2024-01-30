from src.Original.playbar.Ui_play_bar import Ui_Form
from PyQt5 import QtWidgets, QtCore
from typing import Callable, Union
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class PlayBar(QtWidgets.QWidget, Ui_Form):
    """
    Member
    ------
    fps:設定播放的速度\n

    Signal
    -------
    play_signal: 按下播放時會發出的信號
    pause_signal: 按下暫停時會發出的信號
    finish_signal: 播放結束時會發出的信號
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
        update_signal: Callable[[int, bool], bool],
        parent: QtWidgets = None,
        fps: int = 30,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self._fps = fps
        self.can_control = True  # 使用player 自己的Timer

        # parent function
        self.play_signal = play_signal
        self.pause_signal = pause_signal
        self.update_signal = update_signal
        self.finish_signal = finish_signal

        self.play_slider.setTracking(False)  # 只在使用者放掉的時候才會emit value change
        self.timer = QtCore.QTimer()  # 加入定時器
        self.timer.timeout.connect(self.update_slider)  # 設定定時要執行的 function

        self.init_value()
        self.init_bind()

    def init_value(self, total_count: int = 0, now_count: int = 0):
        """初始資料\n
        Param
        --------
        total_count: 所有數量
        now_count: 現在播放到的地方
        """
        self.is_play = False
        self.total_count = total_count
        self.now_count = now_count

        self.play_btn.setText("播放")
        self.play_slider.setMinimum(0)
        self.play_slider.setMaximum(total_count)
        self.play_slider.setValue(self.now_count)

        self.play_btn.setEnabled(True if total_count > 0 else False)
        self.update_frame_label()

    def update_frame_label(self):
        """更新顯示目前播放到的地方的label"""
        self.frame_label.setText(f"{self.play_slider.value()}/{self.total_count}")

    def update_slider(self, new_idx: Union[int, None] = None):
        """更新slider的值
        Param
        ------
        new_idx: slider要跳得位址
        """
        if new_idx:
            if new_idx == 0:  # 強制更新第一張圖
                self.change_frame()
            self.play_slider.setValue(new_idx)
        else:
            self.play_slider.setValue(self.now_count + 1)

    def init_bind(self):
        """signal binding"""
        self.play_btn.clicked.connect(self.play_clicked)
        self.play_slider.valueChanged.connect(self.change_frame)

    def change_frame(self):
        """更新訊號"""
        is_next_frame = self.now_count == self.play_slider.value() - 1
        self.now_count = self.play_slider.value()
        self.update_frame_label()
        self.update_signal(self.play_slider.value(), is_next_frame)
        if self.now_count >= self.total_count:  # finish playing
            self.finish_playing()

    def finish_playing(self):
        """播放結束"""
        self.timer.stop()
        self.is_play = False
        self.play_btn.setText("播放")
        if self.finish_signal:
            self.finish_signal()

    def stop_playing(self):
        """暫停播放"""
        self.timer.stop()
        self.is_play = False
        self.play_btn.setText("播放")
        if self.pause_signal:
            self.pause_signal()

    def start_playing(self):
        """開始播放"""
        if self.can_control:
            self.timer.start(max(0.001, 1000 / self._fps))
        self.is_play = True
        self.play_btn.setText("暫停")
        if self.play_signal:
            self.play_signal()

    def play_clicked(self):
        """使用者點選播放鍵"""
        if self.is_play:  # 暫停
            self.stop_playing()
        else:  # 開始播放
            if self.total_count <= self.now_count or self.total_count <= 0:
                self.replay()
            else:
                self.start_playing()

    def control_slider(self, can_control: bool):
        """是否可拉選slider\n
        Param:
        ------
        can_control (bool):是否可拉選slider
        """
        self.can_control = can_control
        self.play_slider.setEnabled(can_control)

    def control_play_btn(self, can_control: bool):
        """是否可點選播放以及暫停\n
        Param:
        ------
        can_control (bool):否可點選播放以及暫停
        """
        self.play_btn.setEnabled(can_control)

    def replay(self):
        """重新播放"""
        self.now_count = -1
        self.update_slider(0)
        self.start_playing()

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, v):
        self._fps = max(v, 0.001)
