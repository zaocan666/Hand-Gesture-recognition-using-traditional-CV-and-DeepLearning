from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QLabel, QPushButton, QMessageBox, QWidget, QSpinBox
from PyQt5.QtCore import QTimer
import sys
from PyQt5.QtGui import QFont
from PyQt5 import QtGui
import cv2
from cofig import Area_x_start,Area_x_end,Area_y_start,BI_THRESHOLD,BLUR_SIZE
import copy
from extract_hand_video import get_num_mask

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_basic = Ui_basic()
        self.ui_basic.setParent(self)
        self.ui_basic.initUI()

        self.resize(1200, 700)
        self.center()
        self.setWindowTitle('Hand Gesture')
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

class Ui_basic(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.camera_fresh)
        self.start_flag = False
        self.start_first_flag = True

    def initUI(self):
        self.resize(1200, 700)

        video_label = QLabel(parent=self)
        video_label.setText("摄像头：")
        video_label.move(60, 20)

        self.capture_num = QSpinBox(parent=self)
        self.capture_num.setMinimum(0)
        self.capture_num.move(video_label.x()+video_label.width()+10, 20)

        self.open_src_button = QPushButton(parent = self)
        self.open_src_button.setText("打开摄像头")
        self.open_src_button.move(self.capture_num.x()+self.capture_num.width()+10, 20)
        self.open_src_button.pressed.connect(self.open_capture)

        self.recognize_button = QPushButton(parent=self)
        self.recognize_button.setText("开始识别")
        self.recognize_button.move(self.open_src_button.x()+self.open_src_button.width()+30, self.open_src_button.y())
        self.recognize_button.pressed.connect(self.start_recognize)

        self.reset_button = QPushButton(parent=self)
        self.reset_button.setText("重置")
        self.reset_button.move(self.recognize_button.x() + self.recognize_button.width() + 30, self.recognize_button.y())
        self.reset_button.pressed.connect(self.reset)

        self.src_img_area = QLabel(parent=self)  # 摄像头图形显示区域
        self.src_img_area.resize(500, 500)
        self.src_img_area.move(40, self.open_src_button.y()+self.open_src_button.height()+20)

        self.result_img_area = QLabel(parent=self)  # 结果图形显示区域
        self.result_img_area.resize(400, 400)
        self.result_img_area.move(self.src_img_area.x()+self.src_img_area.width()+110, self.src_img_area.y())

        self.result_text_label = QLabel(parent=self)
        self.result_text_label.setText("识别数字：")
        self.result_text_label.setFont(QFont("宋体",20,QFont.Bold))
        self.result_text_label.move(self.result_img_area.x()+self.result_img_area.width()/2-self.result_text_label.width()/2-40, self.result_img_area.y()+self.result_img_area.height()+30)
        self.result_text_label.setVisible(False)

        self.result_num_label = QLabel(parent=self)
        self.result_num_label.setText("0")
        self.result_num_label.setFont(QFont("宋体",20,QFont.Bold))
        self.result_num_label.setStyleSheet("color:red")
        self.result_num_label.move(self.result_text_label.x()+self.result_text_label.width()+50, self.result_text_label.y())
        self.result_num_label.setVisible(False)

    def open_capture(self):
        '''fileName, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          "./",
                                                          "photo(*.jpg *.png *.bmp);;All Files (*)")

        self.src_img=cv2.imread(fileName)
        try:
            self.src_img.shape
        except:
            QMessageBox.warning(self, '提示','图片无法打开',QMessageBox.Yes)
            return'''

        cap_num =self.capture_num.value()
        self.camera=cv2.VideoCapture(cap_num)
        self.camera.set(10, 200)
        if not self.camera.isOpened():
            QMessageBox.warning(self, '提示', '摄像头无法打开', QMessageBox.Yes)
            return
        #if not flag:
        #    QMessageBox.warning(self, '提示', '摄像头无法打开', QMessageBox.Yes)

        self.camera_timer.start(1)

    def camera_fresh(self):
        flag, raw_image = self.camera.read()
        image = copy.deepcopy(raw_image)

        image = cv2.flip(image, 1)
        cv2.rectangle(image, (int(Area_y_start * image.shape[1]), int(Area_x_start * image.shape[0])),
                      (image.shape[1], int(Area_x_end * image.shape[0])), (0, 255, 0), 4)

        self.showImage(self.src_img_area, image)

        if self.start_flag:
            if self.start_first_flag:
                self.MOG2 = cv2.createBackgroundSubtractorMOG2(0, 50)
                self.result_text_label.setVisible(True)
                self.result_num_label.setVisible(True)
                self.result_img_area.setVisible(True)
                self.start_first_flag = False

            mask = self.MOG2.apply(image, learningRate=0)
            count, hand_rgb = get_num_mask(image, mask)
            self.showImage(self.result_img_area, hand_rgb)
            self.result_num_label.setText(str(count))

    def showImage(self, qlabel, img):
        size = (int(qlabel.width()), int(qlabel.height()))
        shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # cv2.imshow('img', shrink)
        shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  QtGui.QImage.Format_RGB888)

        qlabel.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def start_recognize(self):
        if not self.camera_timer.isActive():
            QMessageBox.warning(self, '提示','请先打开原图片',QMessageBox.Yes)
            return
        else:
            self.start_flag=True

    def reset(self):
        self.result_img_area.setVisible(False)
        self.result_num_label.setVisible(False)
        self.result_text_label.setVisible(False)
        self.start_flag=False
        self.start_first_flag=True
        self.MOG2=None

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())
