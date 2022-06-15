import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = np.load('../../datasets/binary_image_data.npy',
                                           allow_pickle=True)
print('X_train shape :', X_train.shape)
print('Y_train shape :', Y_train.shape)
print('X_test shape :', X_test.shape)
print('Y_test shape :', Y_train.shape)

model = Sequential()
model.add(Conv2D(32, input_shape=(64, 64, 3),
                 kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))

model.add(Conv2D(32,
                 kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))

model.add(Conv2D(32,
                 kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) # back을 할 때 학습을 적용해 수정하지 않는다. random하게, 과대적합 방지, back 할 때만 실행
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
               optimizer= 'adam', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=7) # val_loss 폭을 보고 7 epoch 동안 좋아지지 않으면 학습을 중단
fit_hist = model.fit(X_train, Y_train,
                     batch_size=64, epochs=100,
                     validation_split=0.15, callbacks = [early_stopping])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])
model.save('../../models/cat_and_dog_binary_classfication{}.h5'.format(score[1]))# 파일명에 몇 % 짜리 모델인지
plt.plot(fit_hist.history['loss'], label = 'loss')
plt.plot(fit_hist.history['val_loss'], label = 'val_loss')
plt.plot(fit_hist.history['accuracy'], label = 'accuracy')
plt.plot(fit_hist.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()