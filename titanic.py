# -*- coding: utf-8 -*-
"""titanic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q0OAixzBv3dV352KmU4o11GiTDjtZ09Y
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



raw_data = sns.load_dataset('titanic')
print(raw_data)

print(raw_data.isnull().sum())  #age랑 deck가 문제임

#deck컬럼 삭제
 clean_data = raw_data.dropna(axis=1, thresh=500) #none값이500 개 이상인 컬럼삭제
 print(clean_data.columns)

#age는 평균나이로 대체
mean_age = clean_data['age'].mean()
print(mean_age)

clean_data['age'].fillna(mean_age, inplace=True)
print(clean_data.head())

#중복되는 컬럼 삭제
clean_data.drop(['embark_town', 'alive','who'], axis=1, inplace=True)

clean_data.info()
#embarked열에 값이 부족함

#위에 값으로 채워주기
clean_data['embarked'].fillna(
    method='ffill',inplace=True)
#다시 null값 보면 0개 확인
print(clean_data.isnull().sum())

#target인 survived만 빼내기
label=list(clean_data.columns)
keep = label.pop(0)
target = clean_data[[keep]]
training_data = clean_data[label]
print(training_data.head())
print(target.head())

print(keep)
print(label)

training_data = clean_data[label]
target = clean_data[[keep]]

print(training_data)
print(target)

print(training_data[['sex','class']].head())

training_data['sex'].replace({'male':0, 'female':1}, inplace=True)
training_data['class'].replace({'First':1, 'Second':2, 'Third':3}, inplace=True)
training_data['embarked'].replace({'S':1, 'C':2, 'Q':3}, inplace=True)

print(training_data['embarked'].unique())

training_data.info()

print(target[keep].sum())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(training_data)   #numpy의 ndarray가 되어있음 그래서 df로 바꿔줘야함
print(type(scaled_data))
scaled_data = pd.DataFrame(scaled_data, columns=label)
print(scaled_data.head())   #평균이 0이고 표준편차가 1인 정규분포를따르는데이터, 표준정규분포

print(scaled_data.describe().T)   #평균은 0, 표준편차는1

scaled_data.boxplot(column=label,
                    showmeans=True)
plt.show()

from sklearn.model_selection import train_test_split
#데이터중 30프로를 떼서 나중에 테스트용으로 씀
X_train, X_test, Y_train, Y_test = train_test_split(
    scaled_data, target, test_size=0.30)

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('Y_train:', Y_train.shape)
print('Y_test:', Y_test.shape)

model = Sequential()
model.add(Dense(512, input_dim=10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='mse', optimizer='adam',
              metrics=['binary_accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=50, epochs=7,
                     validation_split=0.2, verbose=1)

plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])  #검증정확도

plt.show()

#검증할때 쓰는 evaluate   진행사항 안보려고 verbose=0
score = model.evaluate(X_test, Y_test, verbose=0)  #loss랑 accuracy만 구함 testset에 대한 학습은 하지 않음
print('Keras DNN model loss:', score[0])  #evaluate는 리스트로 값을 반환, 0이 loss, 1이 accuracy
print('Keras DNN model accuracy:', score[1])

plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])  #검증정확도

plt.show()

score = model.evaluate(X_test, Y_test, verbose=0)
print('loss',score[0])
print('accuracy',score[1])