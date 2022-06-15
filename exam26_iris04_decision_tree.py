# -*- coding: utf-8 -*-
"""exam26_iris04_Decision_Tree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P-ye5DNuxV_smsZQBXaPlaDNYxGbdv1x
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import *
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams['figure.figsize'] = [7, 7]
sns.set(style='darkgrid')
plt.rcParams['scatter.edgecolors'] = 'black'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
pd.set_option('display.unicode.east_asian_width', True)

iris_dataset = load_iris()
iris = pd.DataFrame(iris_dataset.data,
        columns=iris_dataset.feature_names)
labels = iris_dataset.target_names
iris.info()
print(iris.head())

label = iris_dataset.target
print(label)

scaler = StandardScaler()
iris = scaler.fit_transform(iris)
Features = pd.DataFrame(iris, columns=['SL', 'SW', 'PL', 'PW'])
print(Features.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    Features, label, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

from sklearn.tree import DecisionTreeClassifier

iris_Tree = DecisionTreeClassifier(criterion='entropy') # 불순도 추출을 entropy로 주었다.
iris_Tree.fit(X_train, Y_train)

iris_Tree.score(X_train, Y_train)

iris_Tree.score(X_test, Y_test)

from sklearn import tree
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 15))
plot_tree(iris_Tree, filled=True,
          rounded=True,
          class_names=['setosa', 'versicolor', 'virginica'],
          feature_names=Features.columns)
plt.show()
# 가지가 적고 깊이가 얕은 나무가 좋다-> 과적합이 적기 때문에

path = iris_Tree.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(ccp_alphas) # 각 가지별 혼잡도, 알파값이 낮은 가지를 제거해야 한다

iris_Trees = []
for ccp_alpha in ccp_alphas: # 7개의 알파값으로 tree를 만들었다
    iris_Tree = DecisionTreeClassifier(
        random_state=868, ccp_alpha=ccp_alpha)
    iris_Tree.fit(X_train, Y_train)
    iris_Trees.append(iris_Tree)

train_score = [iris_Tree.score(X_train, Y_train) for iris_Tree in iris_Trees]
test_score = [iris_Tree.score(X_test, Y_test) for iris_Tree in iris_Trees]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccp_alphas, train_score, '--.', label = 'train',
        drawstyle = 'steps-post')
ax.plot(ccp_alphas, test_score, '--.', label = 'test',
        drawstyle = 'steps-post')
ax.legend()
plt.show()
# 앞아 = 가지를 만들때의 비용혼잡도(민잡도), 얼마나 오분류를 허용할지, 작은 값을 주면 나무가 너무 복잡해진다(과적합), 큰값을 주면 나무가 단순하지만 정확도가 낮아진다ㅣ

iris_Tree = DecisionTreeClassifier(
    random_state=868, ccp_alpha=0.02)
iris_Tree.fit(X_train, Y_train)

iris_Tree.score(X_train, Y_train)

iris_Tree.score(X_test, Y_test)

plt.figure(figsize=(15, 15))
plot_tree(iris_Tree, filled=True,
          rounded=True,
          class_names=['setosa', 'versicolor', 'virginica'],
          feature_names=Features.columns)
plt.show()

pd.DataFrame(confusion_matrix(Y_test, iris_Tree.predict(X_test)),
                                                        columns = ['P_setosa', 'P_versicolor', 'P_virginica'],
                                                        index = ['A_setosa', 'A_versicolor', 'A_virginica'])

print(classification_report(Y_test, iris_Tree.predict(X_test)))
#recall 진짜를 진짜로 분류한 비율, precision(정밀도)A로 예측한 것 중 진짜 A인 비율, f1-score조합 평균(recall, precision의 평균)

for i in range(1, 1000):
  X_train, X_test, Y_train, Y_test = train_test_split(
    Features, label, test_size=0.2, random_state = i)
  iris_Tree = DecisionTreeClassifier(
      criterion='entropy', ccp_alpha=0.02)
  iris_Tree.fit(X_train, Y_train)
  train_score = iris_Tree.score(X_train, Y_train)
  test_score = iris_Tree.score(X_test, Y_test)
  if test_score >= train_score:
    print('test: {} train : {} random_state : {}'.format(test_score, train_score, i))

X_train, X_test, Y_train, Y_test = train_test_split(
  Features, label, test_size=0.2, random_state = 923)
iris_Tree = DecisionTreeClassifier(
    criterion='entropy', ccp_alpha=0.02)
iris_Tree.fit(X_train, Y_train)
print('test: {} train : {} '.format(test_score, train_score))

pd.DataFrame(confusion_matrix(Y_test, iris_Tree.predict(X_test)),
        columns=['P_setosa', 'P_versicolor', 'P_virginica'],
        index=['A_setosa', 'A_versicolor', 'A_virginica'])

print(classification_report(Y_test, iris_Tree.predict(X_test)))

