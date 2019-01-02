from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

# 查看iris数据集
iris = load_iris()
print(iris)

knn = neighbors.KNeighborsClassifier()
# 训练数据集
knn.fit(iris.data, iris.target)
# 预测
predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(predict)
print(knn.predict_proba(X=[[0.1, 0.2, 0.3, 0.4]]))
