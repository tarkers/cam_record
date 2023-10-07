# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Chen\transform_bag\view.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2037, 1118)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.video_2d_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.video_2d_text.setFont(font)
        self.video_2d_text.setText("")
        self.video_2d_text.setObjectName("video_2d_text")
        self.verticalLayout_3.addWidget(self.video_2d_text)
        self.view_2D = QtWidgets.QGraphicsView(self.centralwidget)
        self.view_2D.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view_2D.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view_2D.setObjectName("view_2D")
        self.verticalLayout_3.addWidget(self.view_2D)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.video_3d_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.video_3d_text.setFont(font)
        self.video_3d_text.setText("")
        self.video_3d_text.setObjectName("video_3d_text")
        self.verticalLayout_2.addWidget(self.video_3d_text)
        self.view_3D = QtWidgets.QGraphicsView(self.centralwidget)
        self.view_3D.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view_3D.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view_3D.setObjectName("view_3D")
        self.verticalLayout_2.addWidget(self.view_3D)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.save_btn = QtWidgets.QGroupBox(self.centralwidget)
        self.save_btn.setMinimumSize(QtCore.QSize(250, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.save_btn.setFont(font)
        self.save_btn.setStyleSheet("")
        self.save_btn.setFlat(False)
        self.save_btn.setCheckable(False)
        self.save_btn.setObjectName("save_btn")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.save_btn)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 40, 248, 315))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 1)
        self.load_btn = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.load_btn.setObjectName("load_btn")
        self.gridLayout_2.addWidget(self.load_btn, 0, 0, 1, 2)
        self.video_box = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.video_box.setMinimumSize(QtCore.QSize(150, 0))
        self.video_box.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.video_box.setObjectName("video_box")
        self.gridLayout_2.addWidget(self.video_box, 2, 1, 1, 1)
        self.subject_box = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.subject_box.setMinimumSize(QtCore.QSize(150, 0))
        self.subject_box.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.subject_box.setObjectName("subject_box")
        self.gridLayout_2.addWidget(self.subject_box, 1, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.gridLayoutWidget_2)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 200))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(0, 40, 251, 161))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.wrong_btn = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.wrong_btn.setFont(font)
        self.wrong_btn.setObjectName("wrong_btn")
        self.gridLayout_5.addWidget(self.wrong_btn, 0, 0, 1, 1)
        self.save_btn_2 = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.save_btn_2.setFont(font)
        self.save_btn_2.setObjectName("save_btn_2")
        self.gridLayout_5.addWidget(self.save_btn_2, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_2, 3, 0, 1, 2)
        self.gridLayout.addWidget(self.save_btn, 0, 2, 1, 1)
        self.linechart = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.linechart.sizePolicy().hasHeightForWidth())
        self.linechart.setSizePolicy(sizePolicy)
        self.linechart.setMinimumSize(QtCore.QSize(1600, 300))
        self.linechart.setMaximumSize(QtCore.QSize(1800, 300))
        self.linechart.setObjectName("linechart")
        self.gridLayout.addWidget(self.linechart, 2, 0, 1, 2)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 1, 0, 1, 1)
        self.d_right_box = QtWidgets.QCheckBox(self.centralwidget)
        self.d_right_box.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.d_right_box.setFont(font)
        self.d_right_box.setChecked(True)
        self.d_right_box.setObjectName("d_right_box")
        self.gridLayout_4.addWidget(self.d_right_box, 2, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 3, 0, 1, 1)
        self.v_right_box = QtWidgets.QCheckBox(self.centralwidget)
        self.v_right_box.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.v_right_box.setFont(font)
        self.v_right_box.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.v_right_box.setChecked(True)
        self.v_right_box.setObjectName("v_right_box")
        self.gridLayout_4.addWidget(self.v_right_box, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 6, 1, 1, 1)
        self.d_left_box = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.d_left_box.setFont(font)
        self.d_left_box.setChecked(True)
        self.d_left_box.setObjectName("d_left_box")
        self.gridLayout_4.addWidget(self.d_left_box, 2, 1, 1, 1)
        self.v_left_box = QtWidgets.QCheckBox(self.centralwidget)
        self.v_left_box.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.v_left_box.setFont(font)
        self.v_left_box.setChecked(True)
        self.v_left_box.setObjectName("v_left_box")
        self.gridLayout_4.addWidget(self.v_left_box, 0, 1, 1, 1)
        self.v_left_text = QtWidgets.QLabel(self.centralwidget)
        self.v_left_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.v_left_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.v_left_text.setObjectName("v_left_text")
        self.gridLayout_4.addWidget(self.v_left_text, 1, 1, 1, 1)
        self.d_left_text = QtWidgets.QLabel(self.centralwidget)
        self.d_left_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.d_left_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.d_left_text.setObjectName("d_left_text")
        self.gridLayout_4.addWidget(self.d_left_text, 3, 1, 1, 1)
        self.v_right_text = QtWidgets.QLabel(self.centralwidget)
        self.v_right_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.v_right_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.v_right_text.setObjectName("v_right_text")
        self.gridLayout_4.addWidget(self.v_right_text, 1, 2, 1, 1)
        self.d_right_text = QtWidgets.QLabel(self.centralwidget)
        self.d_right_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.d_right_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.d_right_text.setObjectName("d_right_text")
        self.gridLayout_4.addWidget(self.d_right_text, 3, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 4, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 5, 0, 1, 1)
        self.d_left_box_2d = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.d_left_box_2d.setFont(font)
        self.d_left_box_2d.setChecked(False)
        self.d_left_box_2d.setObjectName("d_left_box_2d")
        self.gridLayout_4.addWidget(self.d_left_box_2d, 4, 1, 1, 1)
        self.d_right_box_2d = QtWidgets.QCheckBox(self.centralwidget)
        self.d_right_box_2d.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.d_right_box_2d.setFont(font)
        self.d_right_box_2d.setChecked(False)
        self.d_right_box_2d.setObjectName("d_right_box_2d")
        self.gridLayout_4.addWidget(self.d_right_box_2d, 4, 2, 1, 1)
        self.d_left_text_2d = QtWidgets.QLabel(self.centralwidget)
        self.d_left_text_2d.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.d_left_text_2d.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.d_left_text_2d.setObjectName("d_left_text_2d")
        self.gridLayout_4.addWidget(self.d_left_text_2d, 5, 1, 1, 1)
        self.d_right_text_2d = QtWidgets.QLabel(self.centralwidget)
        self.d_right_text_2d.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.d_right_text_2d.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.d_right_text_2d.setObjectName("d_right_text_2d")
        self.gridLayout_4.addWidget(self.d_right_text_2d, 5, 2, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_4, 2, 2, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.play_button = QtWidgets.QPushButton(self.centralwidget)
        self.play_button.setObjectName("play_button")
        self.horizontalLayout.addWidget(self.play_button)
        self.frame_slider = QtWidgets.QSlider(self.centralwidget)
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName("frame_slider")
        self.horizontalLayout.addWidget(self.frame_slider)
        self.frame_label = QtWidgets.QLabel(self.centralwidget)
        self.frame_label.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.frame_label.setObjectName("frame_label")
        self.horizontalLayout.addWidget(self.frame_label)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.save_btn.setTitle(_translate("MainWindow", "工作區"))
        self.label_2.setText(_translate("MainWindow", "Video:"))
        self.label.setText(_translate("MainWindow", "Subject:"))
        self.load_btn.setText(_translate("MainWindow", "Load_Subjects"))
        self.groupBox_2.setTitle(_translate("MainWindow", "操作列表"))
        self.wrong_btn.setText(_translate("MainWindow", "Wrong frame"))
        self.save_btn_2.setText(_translate("MainWindow", "Save"))
        self.label_5.setText(_translate("MainWindow", "Angle:"))
        self.d_right_box.setText(_translate("MainWindow", "Right"))
        self.label_3.setText(_translate("MainWindow", "VICON"))
        self.label_4.setText(_translate("MainWindow", "3D Line"))
        self.label_6.setText(_translate("MainWindow", "Angle:"))
        self.v_right_box.setText(_translate("MainWindow", "Right"))
        self.d_left_box.setText(_translate("MainWindow", "Left"))
        self.v_left_box.setText(_translate("MainWindow", "Left"))
        self.v_left_text.setText(_translate("MainWindow", "0"))
        self.d_left_text.setText(_translate("MainWindow", "0"))
        self.v_right_text.setText(_translate("MainWindow", "0"))
        self.d_right_text.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "2D Line"))
        self.label_8.setText(_translate("MainWindow", "Angle:"))
        self.d_left_box_2d.setText(_translate("MainWindow", "Left"))
        self.d_right_box_2d.setText(_translate("MainWindow", "Right"))
        self.d_left_text_2d.setText(_translate("MainWindow", "0"))
        self.d_right_text_2d.setText(_translate("MainWindow", "0"))
        self.play_button.setText(_translate("MainWindow", "Play"))
        self.frame_label.setText(_translate("MainWindow", "0/0"))