import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import time

form_window = uic.loadUiType('./mainWidget.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.path = None
        self.setupUi(self)
        self.model = load_model('../../models/cat_and_dog_binary_classfication0.86.h5')

        self.btn_select.clicked.connect(self.predict_image)

    def predict_image(self):

        capture = cv2.VideoCapture(0) # 카메라 인덱스
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 카메라 해상도
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 카메라 해상도

        flag = True
        while flag:
            ret, frame = capture.read()
            cv2.imshow("VideoFrame", frame)
            time.sleep(0.5)
            print('capture')
            cv2.imwrite('./imgs/capture.png', frame) # ./imgs/capture.png 경로에 이미지 저장

            key = cv2.waitKey(33)
            if key == 27:
                flag = False

            pixmap = QPixmap('./imgs/capture.png')
            self.lbl_image.setPixmap(pixmap)

            try:
                img = Image.open('./imgs/capture.png')
                img = img.convert('RGB')
                img = img.resize((64,64))
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1,64,64,3)
            except:
                  print('error')
            predict_value = self.model.predict(data)
            if predict_value > 0.5:
                self.lbl_predict.setText('이 이미지는 ' +
                   str((predict_value[0][0] * 100).round()) + '% 확률로 Dog입니다.')
            else:
                self.lbl_predict.setText('이 이미지는 ' +
                   str(((1 - predict_value[0][0]) * 100).round()) + '% 확률로 Cat입니다.')
        capture.release()
        cv2.destroyAllWindows()

app = QApplication(sys.argv)
mainWindow = Exam()
mainWindow.show()
sys.exit(app.exec_())