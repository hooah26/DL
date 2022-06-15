# -*- coding: utf-8 -*-
"""exam18_predict_daily_stock_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rd-51zCNL_2sE-dgjjHY5mwHpdGaROvh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

raw_data = pd.read_csv('./datasets/samsung200101_220328.KS.csv')
print(raw_data.head())
raw_data.info()

raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace= True)
print(raw_data.head())

raw_data['Close'].plot()
plt.show()

data_test = raw_data.sort_values('Close')
print(data_test.head())
print(data_test.tail())

data_close = raw_data[['Close']]
data_close.info()
print(data_close.head())

minmaxscaler = MinMaxScaler()
scaled_data = minmaxscaler.fit_transform(data_close)
print(scaled_data[:5])
print(scaled_data.shape)

sequence_X = []
sequence_Y = []

for i in range(len(scaled_data)-30): #30개의 data를 가지고 비교
  x = scaled_data[i:i+30] #scaled_data 30개 중 Close만
  y = scaled_data[i+30] # 30개를 기준으로 그 다음날의 Close
  sequence_X.append(x)
  sequence_Y.append(y)
print(sequence_X[:5])
print(sequence_Y[:5])

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)
print(sequence_X[0])
print(sequence_Y[0])
print(sequence_X.shape)
print(sequence_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    sequence_X, sequence_Y, test_size = 0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(LSTM(50, input_shape=(30,1),
               activation = 'tanh')) #LSTM은 activation으로 tanh, tanh는 -1 ~ 1 사이의 값
model.add(Flatten())
model.add(Dense(1)) # 예측한 값을 그래도 써야 하기 때문에 마지막에는 activation을 사용하지 않는다
model.compile(loss = 'mse', optimizer='adam') # 분류가 아니므로 metrics를 안 쓴다.
model.summary()

fit_hist = model.fit(X_train, Y_train, epochs = 100, 
                     validation_data = (X_test, Y_test), shuffle = False) # 매 epoch 마다 validation_data의 값만큼 뽑아 loss 값만 검증,(X_test, Y_test)에 back를 하지 않아 학습(수정)하지 않는다. 
                     #but forward에서 data 가 많아 학습 효과가 좋다, shuffle = False=>data를 넣는 순서대로 한다.

plt.plot(fit_hist.history['loss'][10:], label = 'loss')
plt.plot(fit_hist.history['val_loss'][10:], label = 'val_loss')
plt.legend()
plt.show()

model.save('./stock_close_predict.h5')

pred = model.predict(X_test)

plt.plot(Y_test[:30], label = 'actual')
plt.plot(pred[:30], label = 'predict')
plt.legend()
plt.show()

last_data_30 = scaled_data[-30:].reshape(1, 30, 1)
today_close = model.predict(last_data_30)

print(today_close)

today_close_value = minmaxscaler.inverse_transform(today_close)
print(today_close_value)

today_actual = 70200
today_actual = np.array(today_actual).reshape(1,1)
scaled_today_actual = minmaxscaler.transform(today_actual) # fit_transfrom -> minmaxscaler가 가지고 있는 min, max 정보가 달라진다.
print(scaled_today_actual)

last_data_29 = scaled_data[-29:]
last_data_30 = np.append(last_data_29,scaled_today_actual)
last_data_30= last_data_30.reshape(1,30,1)
print(last_data_30.shape)

tomorrow_pred = model.predict(last_data_30)
tomorrow_predicted_value = minmaxscaler.inverse_transform(tomorrow_pred)
print('%d'%tomorrow_predicted_value[0][0])
