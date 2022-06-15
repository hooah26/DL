# -*- coding: utf-8 -*-
"""exam08_Introduction_to_machine_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tDsp-JA3gXSVIo5mnCMgJclgwCKwlLqq

#머신러닝 입문

##Linear Regression

기존의 프로그램 방식
"""

def celsius_to_fahrenheit(x):
  return x * 1.8 +32

celsius_value = int(input('섭씨온도를 입력하세요'))
print('화씨온도로', celsius_to_fahrenheit(celsius_value))

"""##머신러닝 방식"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np
import matplotlib.pyplot as plt

data_C = np.array(range(0,100))
data_F = celsius_to_fahrenheit(data_C)
print(data_C)
print(data_F)

model = Sequential()
model.add(InputLayer(input_shape=(1,)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

model.save('./before_learning.h5')

scaled_data_C = data_C / 100
scaled_data_F = data_F / 100
print(scaled_data_C)
print(scaled_data_F)

print(model.predict([0.01]))

fit_hist = model.fit(scaled_data_C, scaled_data_F, epochs=1000) # 학습을 1000 번 실행

model.save('./after_learning.h5')

print(model.predict([0.99]))

"""ax + b 공식의 모델 생성 
처음 모델을 만들 경우 a(kernel) 값은 random, b(bias) 값은 0
"""

plt.plot(fit_hist.history['loss']) # loss = 손실값

# 잡음이 들어가는 겨우
noise = np.array(np.random.normal(0, 0.05, 100)) #평균 0, 표준편차 0.05, 100개 정규분포를 따르는 잡음
print(noise)

noised_scaled_data_F = np.array([])
for data in scaled_data_F:
  noised_scaled_data_F = np.append(
      noised_scaled_data_F, 
      np.random.normal(0, 0.05, 100)+ data) # 정리된 데이터 한개당  100개의 표준편차 noised_data를 더해서 넣는다 그것을 scaled_data_F의 갯수만큼 진행
print(len(noised_scaled_data_F))
print(noised_scaled_data_F[:100])

noised_scaled_data_C = []
for data in range(0, 100): # 0 ~ 100개
  for i in range(100): # 0~100 하나당 
    noised_scaled_data_C.append(data / 100)
noised_scaled_data_C = np.array(noised_scaled_data_C)
print(len(noised_scaled_data_C))
print(noised_scaled_data_C[:101])

plt.scatter(x = noised_scaled_data_C, 
            y = noised_scaled_data_F, alpha = 0.3,
            s = 200)
plt.show()

fig = plt.figure(figsize = (50,50))
ax = fig.add_subplot()
plt.scatter(x = noised_scaled_data_C, 
            y = noised_scaled_data_F, alpha = 0.3,
            s = 200) # x값 하나당 y 값이 100개
plt.show()

model2 = Sequential()
model2.add(InputLayer(input_shape=(1,)))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer='rmsprop')
model2.summary()

print(model2.predict([0.01]))

fit_hist = model2.fit(noised_scaled_data_C, noised_scaled_data_F, epochs = 20)

model2.save('noised_after_learnning.h5')

print(model2.predict([0.01]))

celsius_value = int(input())
print('화씨온도로', model2.predict([celsius_value/100])*100)

