# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Chen\transform_bag\stream.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1219, 1002)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 10, 1041, 581))
        self.graphicsView.setObjectName("graphicsView")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 610, 551, 131))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 40, 521, 81))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.start = QtWidgets.QPushButton(self.gridLayoutWidget)
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
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.camera_btn.setFont(font)
        self.camera_btn.setObjectName("camera_btn")
        self.gridLayout.addWidget(self.camera_btn, 0, 0, 1, 1)
        self.stop = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.stop.setMinimumSize(QtCore.QSize(0, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.stop.setFont(font)
        self.stop.setObjectName("stop")
        self.gridLayout.addWidget(self.stop, 0, 2, 1, 1)
        self.save_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.save_label.setFont(font)
        self.save_label.setObjectName("save_label")
        self.gridLayout.addWidget(self.save_label, 1, 0, 1, 3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 750, 551, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 521, 80))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.square_btn = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.square_btn.setMinimumSize(QtCore.QSize(0, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.square_btn.setFont(font)
        self.square_btn.setObjectName("square_btn")
        self.gridLayout_2.addWidget(self.square_btn, 0, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(580, 610, 471, 311))
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_3)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 441, 271))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_4 = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_4)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 421, 64))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_4.setMaximumSize(QtCore.QSize(30, 15))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.row_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget_2)
        self.row_box.setObjectName("row_box")
        self.horizontalLayout_2.addWidget(self.row_box)
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_5.setMaximumSize(QtCore.QSize(30, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.col_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget_2)
        self.col_box.setObjectName("col_box")
        self.horizontalLayout_2.addWidget(self.col_box)
        self.label_6 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_6.setMaximumSize(QtCore.QSize(30, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        self.mm_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget_2)
        self.mm_box.setObjectName("mm_box")
        self.horizontalLayout_2.addWidget(self.mm_box)
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_4)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 90, 421, 171))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.clip_btn = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clip_btn.sizePolicy().hasHeightForWidth())
        self.clip_btn.setSizePolicy(sizePolicy)
        self.clip_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.clip_btn.setMaximumSize(QtCore.QSize(160, 25))
        self.clip_btn.setMouseTracking(False)
        self.clip_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.clip_btn.setObjectName("clip_btn")
        self.gridLayout_3.addWidget(self.clip_btn, 1, 1, 1, 1)
        self.camera_name = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.camera_name.setObjectName("camera_name")
        self.gridLayout_3.addWidget(self.camera_name, 0, 1, 1, 1)
        self.result_btn = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.result_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.result_btn.setMaximumSize(QtCore.QSize(160, 16777215))
        self.result_btn.setObjectName("result_btn")
        self.gridLayout_3.addWidget(self.result_btn, 5, 1, 1, 1)
        self.video_btn = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_btn.sizePolicy().hasHeightForWidth())
        self.video_btn.setSizePolicy(sizePolicy)
        self.video_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.video_btn.setMaximumSize(QtCore.QSize(160, 25))
        self.video_btn.setMouseTracking(False)
        self.video_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.video_btn.setObjectName("video_btn")
        self.gridLayout_3.addWidget(self.video_btn, 1, 0, 1, 1)
        self.mtx_btn = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mtx_btn.sizePolicy().hasHeightForWidth())
        self.mtx_btn.setSizePolicy(sizePolicy)
        self.mtx_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.mtx_btn.setMaximumSize(QtCore.QSize(160, 25))
        self.mtx_btn.setMouseTracking(False)
        self.mtx_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mtx_btn.setObjectName("mtx_btn")
        self.gridLayout_3.addWidget(self.mtx_btn, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.calibration_btn = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.calibration_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.calibration_btn.setMaximumSize(QtCore.QSize(160, 25))
        self.calibration_btn.setObjectName("calibration_btn")
        self.gridLayout_3.addWidget(self.calibration_btn, 3, 1, 1, 1)
        self.save_btn = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_btn.sizePolicy().hasHeightForWidth())
        self.save_btn.setSizePolicy(sizePolicy)
        self.save_btn.setMinimumSize(QtCore.QSize(0, 25))
        self.save_btn.setMaximumSize(QtCore.QSize(160, 25))
        self.save_btn.setMouseTracking(False)
        self.save_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.save_btn.setObjectName("save_btn")
        self.gridLayout_3.addWidget(self.save_btn, 5, 0, 1, 1)
        self.board_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.board_label.setFont(font)
        self.board_label.setText("")
        self.board_label.setObjectName("board_label")
        self.gridLayout_3.addWidget(self.board_label, 2, 0, 1, 2)
        self.verticalLayout.addWidget(self.groupBox_4)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(30, 890, 531, 41))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.error_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.error_label.setFont(font)
        self.error_label.setText("")
        self.error_label.setObjectName("error_label")
        self.verticalLayout_2.addWidget(self.error_label)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "RECORD"))
        self.start.setText(_translate("MainWindow", "Start Record"))
        self.camera_btn.setText(_translate("MainWindow", "Close Cam"))
        self.stop.setText(_translate("MainWindow", "Stop Record"))
        self.save_label.setText(_translate("MainWindow", "save to:"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Alingment"))
        self.square_btn.setText(_translate("MainWindow", "sqaure_line"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Calibration"))
        self.groupBox_4.setTitle(_translate("MainWindow", "ChessBoard"))
        self.label_4.setText(_translate("MainWindow", "row:"))
        self.label_5.setText(_translate("MainWindow", "col:"))
        self.label_6.setText(_translate("MainWindow", "mm:"))
        self.clip_btn.setText(_translate("MainWindow", "Load Clips"))
        self.camera_name.setText(_translate("MainWindow", "camera_0"))
        self.result_btn.setText(_translate("MainWindow", "See Result"))
        self.video_btn.setText(_translate("MainWindow", "Load Video"))
        self.mtx_btn.setText(_translate("MainWindow", "Load Matrics"))
        self.label.setText(_translate("MainWindow", "camera name:"))
        self.calibration_btn.setText(_translate("MainWindow", "Start Calibration"))
        self.save_btn.setText(_translate("MainWindow", "Save_Data"))
