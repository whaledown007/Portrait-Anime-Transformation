"""
导入各种所需的库和模块，用于图像处理、机器学习、图形用户界面（GUI）开发等功能
"""
from utils import *
import logging
import random
from collections import OrderedDict
import cv2
import imageio
from PIL import Image
import logging
import os
import cv2
import re
import numpy as np
import sys
import random
import datetime
import torch
from utils import *
from torchvision.transforms import transforms
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtMultimedia import QCamera, QCameraInfo, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PIL import Image
import torch.nn as nn
from networks import *


# 推理函数
"""
run_inference 函数负责将输入图像通过生成器模型 genA2B 转换为输出图像，并保存结果。
主要步骤包括加载图像、预处理、推理、后处理和保存结果
"""
def run_inference(genA2B, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    ## 定义转换器：创建一个将Pillow图像转换为转换器to_tensor
    ## to_tensor是torchvision.transforms模块中的一个转换器，用于将Pillow图像（PIL Image）或NumPy数组转换为PyTorch张量（tensor）
    to_tensor = transforms.ToTensor()
    image = to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = genA2B(image)  # 调用模型的前向传递
        fake_image = output[0] if isinstance(output, tuple) else output  # 如果输出是元组，则选择第一个元素

    fake_image = fake_image.detach().cpu().squeeze(0)
    fake_image = (fake_image.numpy().transpose(1, 2, 0) + 1) / 2
    fake_image = (fake_image * 255).astype(np.uint8)
    fake_image = Image.fromarray(fake_image)

    result_path = "./test/results/results.png"
    fake_image.save(result_path)
    return result_path

# 摄像头线程
# CameraThread 类继承自 QThread，用于处理摄像头的图像捕获
class CameraThread(QtCore.QThread):
    image_captured_signal = QtCore.pyqtSignal(QtGui.QImage)
    image_saved_signal = QtCore.pyqtSignal(str) # 新信号，用于发送保存的图片路径

    ## 初始化摄像头(cv2.VideoCapture)
    def __init__(self, parent=None, preview=False):
        super(CameraThread, self).__init__(parent)
        self.camera = cv2.VideoCapture(0)
        self.preview = preview

    ## 运行循环中持续读取摄像头帧并将其转换为 Qt 图像格式
    def run(self):
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                image = self.convert_cv_qt(frame)
                if self.preview:
                    self.image_captured_signal.emit(image)
            QtCore.QThread.msleep(33)  # 约30fps

    ## capture 方法用于捕获并保存当前帧
    def capture(self):
        ret, frame = self.camera.read()
        if ret:
            image = self.convert_cv_qt(frame)
            self.image_captured_signal.emit(image)
            saved_path = f'captured_image_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
            cv2.imwrite(saved_path, frame)  # 保存图片
            self.image_saved_signal.emit(saved_path)  # 发送保存的图片路径
            self.preview = False  # 拍照后停止预览

    ## convert_cv_qt 方法将 OpenCV 图像转换为 Qt 图像格式
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format.scaled(640, 480, QtCore.Qt.KeepAspectRatio)

    ## stop 方法用于释放摄像头资源并停止线程
    def stop(self):
        self.camera.release()
        self.quit()

    def __del__(self):
        self.stop()

# GUI 主窗口类
class MainWindow(QtWidgets.QMainWindow):
    ## 初始化界面 (loadUi) 和生成器模型
    def __init__(self, genA2B, device):
        super(MainWindow, self).__init__()
        uic.loadUi('convert.ui', self)
        self.genA2B = genA2B
        self.device = device
        self.FileButton.clicked.connect(self.select_image)  # 选择文件按钮
        self.ConvertButton.clicked.connect(self.start_conversion)  # 开始转换按钮
        self.selected_image_path = None  # 用来存储选择的图片路径

        # 创建摄像头线程
        self.camera_thread = CameraThread(self)
        self.camera_thread.image_captured_signal.connect(self.display_captured_image)  # 连接信号到槽函数

        # 连接按钮
        self.CameraButton.clicked.connect(self.start_camera)
        self.CaptureButton.clicked.connect(self.capture_photo)

        self.camera_thread.image_captured_signal.connect(self.display_captured_image)
        self.camera_thread.image_saved_signal.connect(self.update_image_path)  # 连接新信号

    def select_image(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")
        if fileName:
            self.display_image(fileName, self.ImageLabel)  # 显示初始图片
            self.selected_image_path = fileName  # 存储图片路径以备后续转换使用

    ## 启动摄像头
    def start_camera(self):
        if not self.camera_thread.isRunning():
            self.camera_thread.preview = True
            self.camera_thread.start()

    ## 捕获摄像头图像
    def capture_photo(self):
        if self.camera_thread.isRunning():
            self.camera_thread.capture()

    ## 更新图像路径并显示图像
    def update_image_path(self, path):
        self.selected_image_path = path  # 更新图片路径
        self.display_image(path, self.ImageLabel)  # 显示新捕获的图片

    def stop_camera(self):
        self.camera_thread.stop()  # 停止摄像头线程

    ## 开始图像转换
    def start_conversion(self):
        # 检查是否已选择图片
        if self.selected_image_path:
            result_image_path = run_inference(self.genA2B, self.selected_image_path, self.device)  # 进行推理
            self.display_image(result_image_path, self.ResultLabel)  # 显示结果图片
        else:
            # 如果没有选择图片，则提示用户
            QtWidgets.QMessageBox.information(self, "提示", "请先选择一张图片或拍摄一张照片。")

    def display_image(self, image_path, label):
        # 显示图片到指定的 QLabel 控件
        pixmap = QtGui.QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def display_captured_image(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.ImageLabel.setPixmap(pixmap)
        self.ImageLabel.setScaledContents(True)
        # 保存路径更新逻辑在 CameraThread 中处理


# 程序入口
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # 加载模型
    device = 'cpu'
    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=32, n_blocks=4, img_size=256, light=True).to(device)
    params = torch.load('results/SelfieToAnime_params_latest.pt', map_location=device)
    genA2B.load_state_dict(params['genA2B'])

    # 启动 GUI
    mainWindow = MainWindow(genA2B, device)
    mainWindow.show()
    sys.exit(app.exec_())
