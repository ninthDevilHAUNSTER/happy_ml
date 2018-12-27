import sys
import os

from data_loader.data_loader import *
import shaobaobaoer_math_lab
from matplotlib import pyplot as plt


X, Y = boy_and_girl_loading_train_data_in_single_vector('shoe size')
X1, Y1 = boy_and_girl_loading_train_data_in_single_vector('height')
X = np.column_stack((X, X1))
from sklearn import svm

# 虽然我很不想做个调参侠，但是这个 svm 的数学公式也太难了点....
s = svm.SVC(kernel='linear', C=1, gamma='auto')
# 最关键的地方还是在这里...调参...啥意思之后再理解
# 道道还是懂得，简单来说就是找到一条线(平面/超平面）让每个点到这个平面的距离最短
result = s.fit(X, Y)
# print(result)
print(X)
print(Y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min) / 100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z)
Z = Z.reshape(xx.shape)
print(Z.__len__())
print(xx.shape)
print(xx)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
print(xx.min())
print(xx.max())
plt.title('SVC with linear kernel')
plt.show()
