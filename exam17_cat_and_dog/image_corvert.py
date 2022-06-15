from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../../datasets/train/'
categories = ['cat', 'dog']
# files = glob.glob(img_dir + 'cat' + '*.jpg')
# print(files)
# exit()

image_w = 64
image_h = 64

pixel = image_h * image_w * 3
X = []
Y = []
files = None
for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '*') # 지정된 경로 안에 해당하는 조건을 만족하는 파일을, glob=>리스트로 경로를 return
    for i, f in enumerate(files): # jpg 파일은 압축파일이다. -> 모델이게 줄 때는 픽셀값으로 줘야 한다. + 사이즈를 맞춰줘야 한다.
        try:
            img = Image.open(f) #pillow 의 Image
            img = img.convert("RGB")
            img = img.resize((image_w, image_h)) # resize 가로, 세로 사이즈를 tuple로 return
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            if i % 300 == 0:
                print(category, ':', f)
        except:
            print(category, i, f)
X = np.array(X)
Y = np.array(Y)

print(X[0])
print(Y[:5])

X = X / 255  # 스케일링, 픽셀 값이 255개, 색의 차이를 256단계 MinMax
Y = Y #/  255 # sigmoid -> 0, 1 두 개의 값만 반환 그런데 그걸 255로 나누면 학습 진행이 안 된다.

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save('../../datasets/binary_image_data.npy', xy)