# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\65904\Desktop\Train_VIT\cam_record\src\Original\playbar\play_bar.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1036, 35)
        Form.setMinimumSize(QtCore.QSize(0, 0))
        Form.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.play_btn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.play_btn.setFont(font)
        self.play_btn.setObjectName("play_btn")
        self.horizontalLayout_2.addWidget(self.play_btn)
        self.play_slider = QtWidgets.QSlider(Form)
        self.play_slider.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.play_slider.setFont(font)
        self.play_slider.setOrientation(QtCore.Qt.Horizontal)
        self.play_slider.setObjectName("play_slider")
        self.horizontalLayout_2.addWidget(self.play_slider)
        self.frame_label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.frame_label.setFont(font)
        self.frame_label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.frame_label.setObjectName("frame_label")
        self.horizontalLayout_2.addWidget(self.frame_label)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.play_btn.setText(_translate("Form", "播放"))
        self.frame_label.setText(_translate("Form", "0/0"))