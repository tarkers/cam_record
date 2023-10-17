# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stream.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1000)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(0, 20, 1861, 821))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(5, 5, 10, 5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.gridLayoutWidget_4)
        self.tabWidget.setMinimumSize(QtCore.QSize(500, 0))
        self.tabWidget.setMaximumSize(QtCore.QSize(600, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(0, 10, 561, 751))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.gridLayout_5.setContentsMargins(5, 2, 2, 2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.groupBox_5 = QtWidgets.QGroupBox(self.gridLayoutWidget_5)
        self.groupBox_5.setMinimumSize(QtCore.QSize(0, 300))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayoutWidget_7 = QtWidgets.QWidget(self.groupBox_5)
        self.gridLayoutWidget_7.setGeometry(QtCore.QRect(10, 20, 541, 257))
        self.gridLayoutWidget_7.setObjectName("gridLayoutWidget_7")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.gridLayoutWidget_7)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.row_box = QtWidgets.QSpinBox(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.row_box.sizePolicy().hasHeightForWidth())
        self.row_box.setSizePolicy(sizePolicy)
        self.row_box.setMinimumSize(QtCore.QSize(100, 30))
        self.row_box.setMaximumSize(QtCore.QSize(120, 16777215))
        self.row_box.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.row_box.setObjectName("row_box")
        self.gridLayout_7.addWidget(self.row_box, 0, 1, 1, 1)
        self.result_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.result_btn.sizePolicy().hasHeightForWidth())
        self.result_btn.setSizePolicy(sizePolicy)
        self.result_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.result_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.result_btn.setFont(font)
        self.result_btn.setObjectName("result_btn")
        self.gridLayout_7.addWidget(self.result_btn, 5, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(150, 30))
        self.label_11.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_7.addWidget(self.label_11, 5, 0, 1, 1)
        self.mtx_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mtx_btn.sizePolicy().hasHeightForWidth())
        self.mtx_btn.setSizePolicy(sizePolicy)
        self.mtx_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.mtx_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.mtx_btn.setFont(font)
        self.mtx_btn.setMouseTracking(False)
        self.mtx_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mtx_btn.setObjectName("mtx_btn")
        self.gridLayout_7.addWidget(self.mtx_btn, 5, 1, 1, 1)
        self.calibration_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibration_btn.sizePolicy().hasHeightForWidth())
        self.calibration_btn.setSizePolicy(sizePolicy)
        self.calibration_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.calibration_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.calibration_btn.setFont(font)
        self.calibration_btn.setObjectName("calibration_btn")
        self.gridLayout_7.addWidget(self.calibration_btn, 4, 3, 1, 1)
        self.clip_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clip_btn.sizePolicy().hasHeightForWidth())
        self.clip_btn.setSizePolicy(sizePolicy)
        self.clip_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.clip_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.clip_btn.setFont(font)
        self.clip_btn.setMouseTracking(False)
        self.clip_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.clip_btn.setObjectName("clip_btn")
        self.gridLayout_7.addWidget(self.clip_btn, 4, 2, 1, 1)
        self.video_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_btn.sizePolicy().hasHeightForWidth())
        self.video_btn.setSizePolicy(sizePolicy)
        self.video_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.video_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.video_btn.setFont(font)
        self.video_btn.setMouseTracking(False)
        self.video_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.video_btn.setObjectName("video_btn")
        self.gridLayout_7.addWidget(self.video_btn, 4, 1, 1, 1)
        self.save_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_btn.sizePolicy().hasHeightForWidth())
        self.save_btn.setSizePolicy(sizePolicy)
        self.save_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.save_btn.setMaximumSize(QtCore.QSize(120, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.save_btn.setFont(font)
        self.save_btn.setMouseTracking(False)
        self.save_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.save_btn.setObjectName("save_btn")
        self.gridLayout_7.addWidget(self.save_btn, 6, 2, 1, 1)
        self.camera_name = QtWidgets.QLineEdit(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_name.sizePolicy().hasHeightForWidth())
        self.camera_name.setSizePolicy(sizePolicy)
        self.camera_name.setMinimumSize(QtCore.QSize(100, 30))
        self.camera_name.setMaximumSize(QtCore.QSize(120, 16777215))
        self.camera_name.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.camera_name.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.camera_name.setObjectName("camera_name")
        self.gridLayout_7.addWidget(self.camera_name, 6, 1, 1, 1)
        self.col_box = QtWidgets.QSpinBox(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.col_box.sizePolicy().hasHeightForWidth())
        self.col_box.setSizePolicy(sizePolicy)
        self.col_box.setMinimumSize(QtCore.QSize(100, 30))
        self.col_box.setMaximumSize(QtCore.QSize(120, 16777215))
        self.col_box.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.col_box.setObjectName("col_box")
        self.gridLayout_7.addWidget(self.col_box, 0, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(150, 30))
        self.label_12.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_7.addWidget(self.label_12, 6, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setMinimumSize(QtCore.QSize(150, 30))
        self.label_10.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_7.addWidget(self.label_10, 4, 0, 1, 1)
        self.mm_box = QtWidgets.QSpinBox(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mm_box.sizePolicy().hasHeightForWidth())
        self.mm_box.setSizePolicy(sizePolicy)
        self.mm_box.setMinimumSize(QtCore.QSize(100, 30))
        self.mm_box.setMaximumSize(QtCore.QSize(120, 16777215))
        self.mm_box.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.mm_box.setObjectName("mm_box")
        self.gridLayout_7.addWidget(self.mm_box, 0, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(150, 30))
        self.label_2.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_7.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setMinimumSize(QtCore.QSize(150, 30))
        self.label_14.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_7.addWidget(self.label_14, 7, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setMinimumSize(QtCore.QSize(150, 30))
        self.label_13.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_7.addWidget(self.label_13, 3, 0, 1, 1)
        self.square_btn = QtWidgets.QPushButton(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.square_btn.sizePolicy().hasHeightForWidth())
        self.square_btn.setSizePolicy(sizePolicy)
        self.square_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.square_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.square_btn.setFont(font)
        self.square_btn.setObjectName("square_btn")
        self.gridLayout_7.addWidget(self.square_btn, 3, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(100, 30))
        self.label_4.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_7.addWidget(self.label_4, 1, 1, 1, 1)
        self.line = QtWidgets.QFrame(self.gridLayoutWidget_7)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_7.addWidget(self.line, 1, 0, 1, 1)
        self.error_label = QtWidgets.QLabel(self.gridLayoutWidget_7)
        self.error_label.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.error_label.setFont(font)
        self.error_label.setText("")
        self.error_label.setObjectName("error_label")
        self.gridLayout_7.addWidget(self.error_label, 7, 1, 1, 3)
        self.line_2 = QtWidgets.QFrame(self.gridLayoutWidget_7)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_7.addWidget(self.line_2, 1, 2, 1, 2)
        self.gridLayout_5.addWidget(self.groupBox_5, 2, 0, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.gridLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_6)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 20, 541, 231))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.save_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.save_label.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.save_label.setFont(font)
        self.save_label.setObjectName("save_label")
        self.gridLayout.addWidget(self.save_label, 2, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.stop = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stop.sizePolicy().hasHeightForWidth())
        self.stop.setSizePolicy(sizePolicy)
        self.stop.setMinimumSize(QtCore.QSize(0, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.stop.setFont(font)
        self.stop.setObjectName("stop")
        self.gridLayout.addWidget(self.stop, 0, 2, 1, 1)
        self.start = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start.sizePolicy().hasHeightForWidth())
        self.start.setSizePolicy(sizePolicy)
        self.start.setMinimumSize(QtCore.QSize(0, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.start.setFont(font)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 0, 1, 1, 1)
        self.camera_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_btn.sizePolicy().hasHeightForWidth())
        self.camera_btn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.camera_btn.setFont(font)
        self.camera_btn.setObjectName("camera_btn")
        self.gridLayout.addWidget(self.camera_btn, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_6, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.gridLayoutWidget_5)
        self.groupBox.setObjectName("groupBox")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 541, 191))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.row_box_3 = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.row_box_3.sizePolicy().hasHeightForWidth())
        self.row_box_3.setSizePolicy(sizePolicy)
        self.row_box_3.setMinimumSize(QtCore.QSize(100, 30))
        self.row_box_3.setMaximumSize(QtCore.QSize(120, 16777215))
        self.row_box_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.row_box_3.setObjectName("row_box_3")
        self.gridLayout_2.addWidget(self.row_box_3, 1, 1, 1, 1)
        self.row_box_4 = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.row_box_4.sizePolicy().hasHeightForWidth())
        self.row_box_4.setSizePolicy(sizePolicy)
        self.row_box_4.setMinimumSize(QtCore.QSize(100, 30))
        self.row_box_4.setMaximumSize(QtCore.QSize(120, 16777215))
        self.row_box_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.row_box_4.setObjectName("row_box_4")
        self.gridLayout_2.addWidget(self.row_box_4, 1, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(150, 30))
        self.label.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 3, 1, 1)
        self.save_btn_2 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_btn_2.sizePolicy().hasHeightForWidth())
        self.save_btn_2.setSizePolicy(sizePolicy)
        self.save_btn_2.setMinimumSize(QtCore.QSize(100, 30))
        self.save_btn_2.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.save_btn_2.setFont(font)
        self.save_btn_2.setMouseTracking(False)
        self.save_btn_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.save_btn_2.setObjectName("save_btn_2")
        self.gridLayout_2.addWidget(self.save_btn_2, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMinimumSize(QtCore.QSize(150, 30))
        self.label_6.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 2, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem2, 3, 1, 1, 1)
        self.video_btn_2 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_btn_2.sizePolicy().hasHeightForWidth())
        self.video_btn_2.setSizePolicy(sizePolicy)
        self.video_btn_2.setMinimumSize(QtCore.QSize(100, 30))
        self.video_btn_2.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.video_btn_2.setFont(font)
        self.video_btn_2.setMouseTracking(False)
        self.video_btn_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.video_btn_2.setObjectName("video_btn_2")
        self.gridLayout_2.addWidget(self.video_btn_2, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMinimumSize(QtCore.QSize(150, 30))
        self.label_5.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.tab_2)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(0, 0, 691, 781))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox = QtWidgets.QCheckBox(self.gridLayoutWidget_3)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_3.addWidget(self.checkBox, 2, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem3, 3, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_3.addWidget(self.pushButton_2, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_4.addWidget(self.tabWidget, 0, 0, 3, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.gridLayoutWidget_4)
        self.graphicsView_2.setMinimumSize(QtCore.QSize(700, 0))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout_4.addWidget(self.graphicsView_2, 0, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 30))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.horizontalSlider = QtWidgets.QSlider(self.gridLayoutWidget_4)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(0, 20))
        self.horizontalSlider.setMaximum(1000)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.gridLayout_4.addLayout(self.horizontalLayout, 1, 2, 1, 1)
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.gridLayoutWidget_4)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(500, 0))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.gridLayout_4.addWidget(self.graphicsView_3, 0, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_5.setTitle(_translate("MainWindow", "影像校正"))
        self.result_btn.setText(_translate("MainWindow", "顯示結果"))
        self.label_11.setText(_translate("MainWindow", "顯示結果:"))
        self.mtx_btn.setText(_translate("MainWindow", "載入校正參數"))
        self.calibration_btn.setText(_translate("MainWindow", "開始校正"))
        self.clip_btn.setText(_translate("MainWindow", "載入影像集"))
        self.video_btn.setText(_translate("MainWindow", "載入影片"))
        self.save_btn.setText(_translate("MainWindow", "儲存結果"))
        self.camera_name.setText(_translate("MainWindow", "camera"))
        self.camera_name.setPlaceholderText(_translate("MainWindow", "請輸入相機名稱"))
        self.label_12.setText(_translate("MainWindow", "儲存結果:"))
        self.label_10.setText(_translate("MainWindow", "校正影片/影像:"))
        self.label_2.setText(_translate("MainWindow", "校正板參數(row,col,mm):"))
        self.label_14.setText(_translate("MainWindow", "錯誤訊息:"))
        self.label_13.setText(_translate("MainWindow", "水平/垂直定位:"))
        self.square_btn.setText(_translate("MainWindow", "顯示方框"))
        self.label_4.setText(_translate("MainWindow", "處理影像"))
        self.groupBox_6.setTitle(_translate("MainWindow", "直播錄製"))
        self.save_label.setText(_translate("MainWindow", "存於:"))
        self.stop.setText(_translate("MainWindow", "停止錄製"))
        self.start.setText(_translate("MainWindow", "開始錄製"))
        self.camera_btn.setText(_translate("MainWindow", "開啟直播"))
        self.groupBox.setTitle(_translate("MainWindow", "影片剪輯"))
        self.label.setText(_translate("MainWindow", "裁減片段(Start,End):"))
        self.save_btn_2.setText(_translate("MainWindow", "儲存影片"))
        self.label_6.setText(_translate("MainWindow", "儲存影片:"))
        self.video_btn_2.setText(_translate("MainWindow", "載入影片"))
        self.label_5.setText(_translate("MainWindow", "載入影片:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "影片處理"))
        self.checkBox.setText(_translate("MainWindow", "顯示boundingbox"))
        self.pushButton_2.setText(_translate("MainWindow", "載入影片"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "骨架分析"))
        self.pushButton.setText(_translate("MainWindow", "播放"))
