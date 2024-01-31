from src.Original.canvas.Ui_canvas import Ui_Form

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage
from typing import Callable
import cv2
import numpy as np


class Canvas(QtWidgets.QWidget, Ui_Form):
    """
    Member
    ------
    aspect_ratio: 是否遵循圖片的大小比例\n

    Signal
    --------
    clicked_signal: 點選圖片發出(x,y)位置
        @param
        [x,y] (list): 點選x,y的位置
    """

    def __init__(
        self,
        clicked_signal: Callable[[list], None] = None,
        aspect_ratio: bool = True,
        parent: QtWidgets = None,
    ):
        super().__init__(parent)
        self.setupUi(self)

        self._aspect_ratio = aspect_ratio
        self._view_type = ""  # 播放類型
        self.clicked_signal = clicked_signal

        self.iw: int = 0
        self.ih: int = 0
        self.w: int = 0
        self.h: int = 0
        if self.clicked_signal is not None:
            self.scene_label.mousePressEvent = self.getPos
        self.init_value(None)

    def init_value(self, image=None):
        """初始資料\n
        Param
        --------
        image: 展示圖片
        """
        self.scene_label.clear()
        if image:
            self.show_image(image)

    def show_image(self, image:np.ndarray):
        """顯示圖片於畫面上"""
        image = image.copy()

        ## 顯示正在錄影 ##
        if self._view_type == "is_view_record":
            self.image_signal_text(image, "Recorded Video")
        elif self._view_type == "is_record":
            self.image_signal_text(image, "Recording....")

        ih, iw, _ = image.shape
        self.iw = iw
        self.ih = ih

        if self._aspect_ratio:
            h, w = (
                self.size().height() - 5,
                self.size().width() - 5,
            )

            if (ih / h) > (iw / w):
                w, h = h * iw // ih, h
            else:
                w, h = w, w * ih // iw
        else:
            h, w = ih, iw

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        bytesPerline = 3 * w
        self.w, self.h = w, h
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.scene_label.setPixmap(QPixmap.fromImage(qImg))

    def image_signal_text(
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
        """
        圖片加入文字
        """
        image = cv2.putText(
            image, text, place, font, fontScale, color, thickness, lineType
        )

    def getPos(self, event):
        """
        畫布取得點選位置
        """
        x = event.pos().x()
        y = event.pos().y()
        if event.buttons() & QtCore.Qt.LeftButton:
            # print("clicked")
            self.last_mouse_point = self.pixel_to_img_coordinate(x, y)
        elif event.buttons() & QtCore.Qt.RightButton:
            pass

    def pixel_to_img_coordinate(self, x, y):
        """畫布點選位置還原成原圖的位置\n
        Param
        --------
        x: 畫布x
        y: 畫布y\n
        Emit
        -------
        clicked_signal: (x,y) of original image
        """
        ih, iw = self.ih, self.iw
        h, w = self.h, self.w
        if self.clicked_signal:
            self.clicked_signal([int(x * iw // w), int(y * ih // h)])

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @aspect_ratio.setter
    def set_aspect_ratio(self, v):
        self._aspect_ratio = v

    @property
    def view_type(self):
        return self._aspect_ratio

    @view_type.setter
    def view_type(self, v):
        self._view_type = v
