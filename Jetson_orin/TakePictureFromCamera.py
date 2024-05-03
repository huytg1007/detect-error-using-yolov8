import os
import sys
import cv2
import numpy as np
import imutils
import time
import torch
import json


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ftplib import FTP
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from pypylon import pylon


torch.cuda.set_device(0)  # Set to your desired GPU number
cwd = os.getcwd()

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

yellow = [255, 255, 255]  # Yellow in BGR Color Space
# cognex's config

size = (640, 480)
count = 0

is_continue_taking_picture = False
is_take_a_picture = False
prev_time = 0
interval = 2
pTime = 0

NG_time_1 = 0
NG_time_2 = 0

threshold_value = 150
contrast_value = 1
brightness_value = 1
accuracy_value = 20
path = "D:\\Honda_PlusVN\\Python\\Image\\Dataset"
rbg_value = "255, 255, 255"
selected_error = 0
current_pic_1 = ""
current_pic_2 = ""


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1803, 859)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lb_Image_1 = QtWidgets.QLabel(self.centralwidget)
        self.lb_Image_1.setGeometry(QtCore.QRect(350, 20, 711, 511))
        self.lb_Image_1.setStyleSheet("border: 3px solid blue")
        self.lb_Image_1.setText("")
        self.lb_Image_1.setPixmap(
            QtGui.QPixmap(
                "VideoToPicture/EyeDropBottle_Dataset/Dataset Lỗi Miệng Trắng/OK/error_14-03-41.jpg"
            )
        )
        self.lb_Image_1.setObjectName("lb_Image_1")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(360, 660, 341, 141))
        self.groupBox.setObjectName("groupBox")
        self.image_save_log_2 = QtWidgets.QListWidget(self.groupBox)
        self.image_save_log_2.setGeometry(QtCore.QRect(10, 30, 321, 101))
        self.image_save_log_2.setObjectName("image_save_log_2")
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(100, 20, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_stop.setFont(font)
        self.btn_stop.setStyleSheet("background: rgb(255, 107, 107)")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setGeometry(QtCore.QRect(10, 20, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_start.setFont(font)
        self.btn_start.setStyleSheet("background:rgb(56, 209, 255)")
        self.btn_start.setObjectName("btn_start")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 80, 331, 571))
        self.groupBox_3.setObjectName("groupBox_3")
        self.slider_threshold = QtWidgets.QSlider(self.groupBox_3)
        self.slider_threshold.setGeometry(QtCore.QRect(100, 80, 131, 22))
        self.slider_threshold.setMaximum(255)
        self.slider_threshold.setProperty("value", 150)
        self.slider_threshold.setOrientation(QtCore.Qt.Horizontal)  # type: ignore
        self.slider_threshold.setObjectName("slider_threshold")
        self.slider_contrast = QtWidgets.QSlider(self.groupBox_3)
        self.slider_contrast.setGeometry(QtCore.QRect(100, 120, 131, 22))
        self.slider_contrast.setMaximum(127)
        self.slider_contrast.setProperty("value", 1)
        self.slider_contrast.setOrientation(QtCore.Qt.Horizontal)  # type: ignore
        self.slider_contrast.setObjectName("slider_contrast")
        self.btn_choice_color = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_choice_color.setGeometry(QtCore.QRect(150, 190, 75, 23))
        self.btn_choice_color.setObjectName("btn_choice_color")
        self.spin_contrast = QtWidgets.QSpinBox(self.groupBox_3)
        self.spin_contrast.setGeometry(QtCore.QRect(270, 120, 42, 22))
        self.spin_contrast.setMaximum(127)
        self.spin_contrast.setProperty("value", 1)
        self.spin_contrast.setObjectName("spin_contrast")
        self.slider_brightness = QtWidgets.QSlider(self.groupBox_3)
        self.slider_brightness.setGeometry(QtCore.QRect(100, 160, 131, 22))
        self.slider_brightness.setMaximum(100)
        self.slider_brightness.setProperty("value", 1)
        self.slider_brightness.setOrientation(QtCore.Qt.Horizontal)  # type: ignore
        self.slider_brightness.setObjectName("slider_brightness")
        self.txt_RBG = QtWidgets.QLineEdit(self.groupBox_3)
        self.txt_RBG.setGeometry(QtCore.QRect(60, 230, 91, 20))
        self.txt_RBG.setObjectName("txt_RBG")
        self.spin_brightness = QtWidgets.QSpinBox(self.groupBox_3)
        self.spin_brightness.setGeometry(QtCore.QRect(270, 160, 42, 22))
        self.spin_brightness.setMaximum(100)
        self.spin_brightness.setProperty("value", 1)
        self.spin_brightness.setObjectName("spin_brightness")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(20, 80, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(20, 160, 71, 16))
        self.label_6.setObjectName("label_6")
        self.spin_threshold = QtWidgets.QSpinBox(self.groupBox_3)
        self.spin_threshold.setGeometry(QtCore.QRect(270, 80, 42, 22))
        self.spin_threshold.setMaximum(255)
        self.spin_threshold.setProperty("value", 150)
        self.spin_threshold.setObjectName("spin_threshold")
        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setGeometry(QtCore.QRect(20, 230, 21, 16))
        self.label_11.setObjectName("label_11")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(20, 120, 71, 16))
        self.label_5.setObjectName("label_5")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(20, 200, 121, 16))
        self.label_9.setObjectName("label_9")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 480, 331, 71))
        self.groupBox_4.setObjectName("groupBox_4")
        self.cb_from_image = QtWidgets.QComboBox(self.groupBox_4)
        self.cb_from_image.setGeometry(QtCore.QRect(50, 30, 69, 22))
        self.cb_from_image.setObjectName("cb_from_image")
        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setGeometry(QtCore.QRect(10, 30, 47, 13))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_4)
        self.label_8.setGeometry(QtCore.QRect(140, 30, 31, 16))
        self.label_8.setObjectName("label_8")
        self.cb_to_image = QtWidgets.QComboBox(self.groupBox_4)
        self.cb_to_image.setGeometry(QtCore.QRect(170, 30, 69, 22))
        self.cb_to_image.setObjectName("cb_to_image")
        self.cb_list_error = QtWidgets.QComboBox(self.groupBox_3)
        self.cb_list_error.setGeometry(QtCore.QRect(110, 30, 201, 22))
        self.cb_list_error.setObjectName("cb_list_error")
        self.cb_list_error.addItem("")
        self.cb_list_error.addItem("")
        self.cb_list_error.addItem("")
        self.cb_list_error.addItem("")
        self.cb_list_error.addItem("")
        self.cb_list_error.addItem("")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(20, 30, 71, 16))
        self.label_10.setObjectName("label_10")
        self.lb_Image_2 = QtWidgets.QLabel(self.centralwidget)
        self.lb_Image_2.setGeometry(QtCore.QRect(1080, 20, 711, 511))
        self.lb_Image_2.setStyleSheet("border: 3px solid blue")
        self.lb_Image_2.setText("")
        self.lb_Image_2.setPixmap(
            QtGui.QPixmap(
                "VideoToPicture/EyeDropBottle_Dataset/Dataset Lỗi Miệng Trắng/NG/error_13-51-26.jpg"
            )
        )
        self.lb_Image_2.setObjectName("lb_Image_2")
        self.btn_take_picture = QtWidgets.QPushButton(self.centralwidget)
        self.btn_take_picture.setGeometry(QtCore.QRect(190, 20, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_take_picture.setFont(font)
        self.btn_take_picture.setStyleSheet("background: rgb(135, 255, 70)")
        self.btn_take_picture.setObjectName("btn_take_picture")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(740, 660, 341, 141))
        self.groupBox_5.setObjectName("groupBox_5")
        self.error_log = QtWidgets.QListWidget(self.groupBox_5)
        self.error_log.setGeometry(QtCore.QRect(10, 30, 321, 101))
        self.error_log.setObjectName("error_log")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 660, 331, 141))
        self.groupBox_2.setObjectName("groupBox_2")
        self.txt_width = QtWidgets.QLineEdit(self.groupBox_2)
        self.txt_width.setGeometry(QtCore.QRect(50, 80, 51, 20))
        self.txt_width.setObjectName("txt_width")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 80, 31, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(130, 80, 31, 16))
        self.label_3.setObjectName("label_3")
        self.txt_height = QtWidgets.QLineEdit(self.groupBox_2)
        self.txt_height.setGeometry(QtCore.QRect(180, 80, 51, 20))
        self.txt_height.setObjectName("txt_height")
        self.txt_path = QtWidgets.QLineEdit(self.groupBox_2)
        self.txt_path.setGeometry(QtCore.QRect(90, 40, 231, 20))
        self.txt_path.setObjectName("txt_path")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(10, 40, 71, 16))
        self.label.setObjectName("label")

        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(20, 280, 51, 21))
        self.label_12.setObjectName("label_12")
        self.slider_accuracy = QtWidgets.QSlider(self.groupBox_3)
        self.slider_accuracy.setGeometry(QtCore.QRect(80, 280, 141, 22))
        self.slider_accuracy.setMinimum(1)
        self.slider_accuracy.setMaximum(100)
        self.slider_accuracy.setProperty("value", 20)
        self.slider_accuracy.setOrientation(QtCore.Qt.Horizontal)  # type: ignore
        self.slider_accuracy.setObjectName("slider_accuracy")
        self.spin_accuracy = QtWidgets.QSpinBox(self.groupBox_3)
        self.spin_accuracy.setGeometry(QtCore.QRect(240, 280, 42, 22))
        self.spin_accuracy.setMinimum(1)
        self.spin_accuracy.setMaximum(100)
        self.spin_accuracy.setProperty("value", 20)
        self.spin_accuracy.setObjectName("spin_accuracy")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(290, 280, 21, 21))
        self.label_13.setObjectName("label_13")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1803, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Worker = Worker1(MainWindow)
        self.Worker.start()
        self.Worker.ImageUpdate.connect(self.ImageUpdateSlot)

        self.spin_threshold.valueChanged.connect(self.set_threshold_value)
        self.spin_contrast.valueChanged.connect(self.set_contrast_value)
        self.spin_brightness.valueChanged.connect(self.set_brightness_value)
        self.spin_accuracy.valueChanged.connect(self.set_accuracy_value)

        self.slider_threshold.valueChanged.connect(self.set_threshold_value)
        self.slider_contrast.valueChanged.connect(self.set_contrast_value)
        self.slider_brightness.valueChanged.connect(self.set_brightness_value)
        self.slider_accuracy.valueChanged.connect(self.set_accuracy_value)

        self.txt_RBG.textChanged.connect(self.set_rbg)
        self.txt_RBG.textChanged.connect(self.set_rbg)
        self.txt_path.textChanged.connect(self.set_path)

        self.btn_start.clicked.connect(self.on_click_start)
        self.btn_stop.clicked.connect(self.on_click_stop)
        self.btn_take_picture.clicked.connect(self.on_click_take_picture)
        self.btn_choice_color.clicked.connect(self.on_click)

        self.cb_list_error.currentIndexChanged.connect(self.on_select_error_changed)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Log Image Saved"))
        self.btn_stop.setText(_translate("MainWindow", "Stop"))
        self.btn_start.setText(_translate("MainWindow", "Start"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Setting"))
        self.btn_choice_color.setText(_translate("MainWindow", "Choice"))
        self.txt_RBG.setText(_translate("MainWindow", "255, 255, 255"))
        self.label_4.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" font-size:10pt;">Threshold</span></p><p><br/></p></body></html>',
            )
        )
        self.label_6.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" font-size:10pt;">Brightness</span></p></body></html>',
            )
        )
        self.label_11.setText(
            _translate("MainWindow", "<html><head/><body><p>RBG</p></body></html>")
        )
        self.label_5.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" font-size:10pt;">Contrast</span></p><p><br/></p></body></html>',
            )
        )
        self.label_9.setText(
            _translate(
                "MainWindow",
                "<html><head/><body><p>Choice Colo To Detect</p></body></html>",
            )
        )
        self.groupBox_4.setTitle(_translate("MainWindow", "Choice Image To Convert"))
        self.label_7.setText(_translate("MainWindow", "From"))
        self.label_8.setText(_translate("MainWindow", "To"))
        self.cb_list_error.setItemText(0, _translate("MainWindow", "Black Point"))
        self.label_10.setText(
            _translate(
                "MainWindow", "<html><head/><body><p>Choice Error</p></body></html>"
            )
        )
        self.label_12.setText(
            _translate("MainWindow", "<html><head/><body><p>Accuracy</p></body></html>")
        )
        self.label_13.setText(
            _translate("MainWindow", "<html><head/><body><p>%</p></body></html>")
        )
        self.btn_take_picture.setText(_translate("MainWindow", "Take a picture"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Log Error"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Save Image"))
        self.label_2.setText(_translate("MainWindow", "Width"))
        self.label_3.setText(_translate("MainWindow", "Height"))
        self.txt_path.setText(
            _translate(
                "MainWindow",
                "D:\\Honda_PlusVN\\Python\\Image\\Dataset\\ChaiTo\\Lỗi Dị Vật",
            )
        )
        self.label.setText(_translate("MainWindow", "Path To Save"))

    @pyqtSlot()
    def openColorDialog(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # print(color.name())
            self.set_rbg(
                f"{str(color.red())}, {str(color.blue())}, {str(color.green())}"
            )

    def on_click(self):
        self.openColorDialog()  # type: ignore

    def on_click_start(self):
        global is_continue_taking_picture
        is_continue_taking_picture = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def on_click_stop(self):
        global is_continue_taking_picture
        is_continue_taking_picture = False
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)

    def on_click_take_picture(self):
        global is_take_a_picture
        is_take_a_picture = True

    def on_select_error_changed(self, index):
        global threshold_value, contrast_value, brightness_value
        global selected_error
        selected_text = self.cb_list_error.currentText()
        selected_error = index
        if index == 0:
            self.spin_threshold.setValue(219)
            self.slider_threshold.setValue(219)
            threshold_value = 150
            self.spin_contrast.setValue(1)
            self.slider_contrast.setValue(1)
            contrast_value = 4
            self.spin_brightness.setValue(66)
            self.slider_brightness.setValue(66)
            brightness_value = 1

        elif index in [1, 3]:
            self.spin_threshold.setValue(130)
            self.slider_threshold.setValue(130)
            threshold_value = 130
            self.spin_contrast.setValue(2)
            self.slider_contrast.setValue(2)
            contrast_value = 2
            self.spin_brightness.setValue(50)
            self.slider_brightness.setValue(50)
            brightness_value = 50

        elif index == 2:
            self.spin_threshold.setValue(248)
            self.slider_threshold.setValue(248)
            threshold_value = 248
            self.spin_contrast.setValue(1)
            self.slider_contrast.setValue(1)
            contrast_value = 1
            self.spin_brightness.setValue(68)
            self.slider_brightness.setValue(68)
            brightness_value = 68

    def ImageUpdateSlot(self, Image1, Image2):
        self.lb_Image_1.setPixmap(QPixmap.fromImage(Image1))
        self.lb_Image_2.setPixmap(QPixmap.fromImage(Image2))

    def set_threshold_value(self, value):
        self.spin_threshold.setValue(value)
        self.slider_threshold.setValue(value)
        global threshold_value
        threshold_value = value

    def set_contrast_value(self, value):
        self.spin_contrast.setValue(value)
        self.slider_contrast.setValue(value)
        global contrast_value
        contrast_value = value

    def set_brightness_value(self, value):
        self.spin_brightness.setValue(value)
        self.slider_brightness.setValue(value)
        global brightness_value
        brightness_value = value

    def set_accuracy_value(self, value):
        self.spin_accuracy.setValue(value)
        self.slider_accuracy.setValue(value)
        global accuracy_value
        accuracy_value = value

    def set_rbg(self, value):
        self.txt_RBG.setText(str(value))
        global rbg_value
        rbg_value = value

    def set_path(self, value):
        self.txt_path.setText(str(value))
        global path
        path = value

    @QtCore.pyqtSlot()
    def writeSavedImage(self, value):
        self.image_save_log_2.addItem(value)

    def CancelFeed(self):
        self.Worker.stop()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage, QImage)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def run(self):
        global prev_time
        global interval
        global pTime
        global NG_time_1, NG_time_2
        global threshold_value, contrast_value, brightness_value, rbg_value, accuracy_value, path
        global is_continue_taking_picture, is_take_a_picture
        global current_pic_1, current_pic_2

        self.ThreadActive = True
        while self.ThreadActive:
            grabResult = camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )

            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Resize the image
                resize_image = cv2.resize(img, (1000, 800))

                # resize_image = img
                threshold_value = float(threshold_value)

                contrast = float(contrast_value)  # Contrast control ( 0 to 127)
                brightness = float(brightness_value)  # Brightness control (0-100)
                bright = cv2.addWeighted(
                    resize_image, contrast, resize_image, 0, brightness
                )

                _, threshold_image = cv2.threshold(
                    bright, threshold_value, 200, cv2.THRESH_BINARY_INV
                )

                parts = rbg_value.split(", ")
                red = parts[0]
                blue = parts[1]
                green = parts[2]

                b1 = np.array([red, blue, green], dtype=np.uint8)
                b2 = np.array([red, blue, green], dtype=np.uint8)
                bin = cv2.inRange(bright, b1, b2)
                cv2.bitwise_not(bin, bin)

                cnts = cv2.findContours(
                    bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cnts = imutils.grab_contours(cnts)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

                current_datetime = datetime.now()
                curr_time = time.time()
                elapsed_time = curr_time - prev_time

                if elapsed_time > interval and is_continue_taking_picture:
                    try:
                        current_time = current_datetime.strftime("%H-%M-%S")
                        print(f"take a picture {current_time}")
                        cv2.imwrite(f"{path}\\NG_Image_{current_time}.jpg", bright)

                        prev_time = curr_time
                    except Exception as e:
                        print(e)

                if is_take_a_picture:
                    try:
                        current_time = current_datetime.strftime("%H-%M-%S")
                        print(f"take a picture {current_time}")
                        cv2.imwrite(f"{path}\\NG_Image_{current_time}.jpg", bright)

                        # self.main_window.writeSavedImage(self, path)
                        prev_time = curr_time
                        is_take_a_picture = False

                    except Exception as e:
                        print(e)

                # Calculate the elapsed time since the last save
                fps = 1 / (
                    curr_time - pTime
                )  # tính fps Frames per second - đây là  chỉ số khung hình trên mỗi giây
                pTime = curr_time

                cv2.putText(
                    bright,
                    f"1. FPS: {int(fps)}",
                    (0, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 255, 0),
                    3,
                )

                if selected_error == 0:
                    try:
                        FlippedImage = bright
                        ConvertToQtFormat = QImage(
                            FlippedImage.data,
                            FlippedImage.shape[1],
                            FlippedImage.shape[0],
                            QImage.Format_RGB888,
                        )
                        Pic1 = ConvertToQtFormat.scaled(711, 600, Qt.KeepAspectRatio)

                        FlippedImage2 = threshold_image
                        ConvertToQtFormat2 = QImage(
                            FlippedImage2.data,
                            FlippedImage.shape[1],
                            FlippedImage.shape[0],
                            QImage.Format_RGB888,
                        )
                        Pic2 = ConvertToQtFormat2.scaled(711, 600, Qt.KeepAspectRatio)

                        current_pic_1 = "bright"
                        current_pic_2 = "threshold_image"
                        self.ImageUpdate.emit(Pic1, Pic2)
                    except Exception as e:
                        print(e)

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
