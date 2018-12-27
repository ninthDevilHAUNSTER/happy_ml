'''高斯朴素贝叶斯'''
import sys, os

sys.path.append('./')
import numpy as np
from sklearn import naive_bayes
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import pandas as pd
import random

import singal_vector as single_vector
import multi_vector

# from singal_vector import *
# from multi_vector import *

# global variables
header = ['height', 'weight', 'shoe size']


# 1、处理数据：从csv文件中载入数据，然后划分为训练集和测试集。

# height(cm) | weight(kg) | shoe size(EUR)
def loading_train_data():
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('./dates/boy.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    # boy_data.insert(header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('./dates/girl.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    # girl_data.insert(header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_lables = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_lables


def loading_train_data_in_random_size(size):
    boy_data = pd.read_csv('./dates/boy.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    # boy_data.insert(header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('./dates/girl.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    # girl_data.insert(header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)

    a = boy_data
    b = np.array([1] * boy_data.__len__())

    li = list(range(size))
    random.shuffle(li)
    c = []
    d = []
    for i in li:
        c.append(a[i])
        d.append(b[i])

    boy_data = c

    a = girl_data
    b = np.array([0] * boy_data.__len__())

    li = list(range(size))
    random.shuffle(li)
    c = []
    d = []
    for i in li:
        c.append(a[i])
        d.append(b[i])

    girl_data = c

    result_mat = np.vstack((boy_data, girl_data))
    class_lables = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())

    return result_mat, class_lables


def loading_test_data():
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('./dates/boynew.txt', names=['height', 'weight', 'shoe size'], sep=' ')
    # boy_data.insert(header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('./dates/girlnew.txt', names=['height', 'weight', 'shoe size'], sep=' ')
    # girl_data.insert(header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_label = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_label


def loading_train_data_in_single_vector(vector_name):
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('./dates/boy.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    girl_data = pd.read_csv('./dates/girl.txt', names=['height', 'weight', 'shoe size'], sep='\t')

    for i in header:
        if vector_name != i:
            boy_data.pop(i)
            girl_data.pop(i)

    girl_data = np.array(girl_data.values)
    boy_data = np.array(boy_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_lables = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_lables


def loading_test_data_in_single_vector(vector_name):
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('./dates/boynew.txt', names=['height', 'weight', 'shoe size'], sep=' ')
    girl_data = pd.read_csv('./dates/girlnew.txt', names=['height', 'weight', 'shoe size'], sep=' ')

    for i in header:
        if vector_name != i:
            boy_data.pop(i)
            girl_data.pop(i)

    girl_data = np.array(girl_data.values)
    boy_data = np.array(boy_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_label = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_label


def MyGaussianNB(trainMat=np.array([]), Classlabels=np.array([]), testDoc=np.array([])):
    # -----sklearn GaussianNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 高斯分布
    clf = naive_bayes.GaussianNB()
    clf.set_params(priors=[0.6776119402985075, 0.32238805970149254])
    # 设置先验概率
    clf.fit(X, Y)
    y_pred = clf.predict(testDoc)
    return y_pred


def MyMultinomialNB(trainMat=np.array([]), Classlabels=np.array([]), testDoc=np.array([])):
    # -----sklearn GaussianNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 高斯分布
    clf = naive_bayes.MultinomialNB()
    clf.set_params()
    # 设置先验概率
    clf.fit(X, Y)
    # 测试预测结果
    # clf.fit(trainMat, Classlabels)
    y_pred = clf.predict(testDoc)
    return y_pred


def MyBernoulliNB(trainMat=np.array([]), Classlabels=np.array([]), testDoc=np.array([])):
    # -----sklearn GaussianNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 高斯分布
    clf = naive_bayes.BernoulliNB()
    clf.set_params()
    # 设置先验概率
    clf.fit(X, Y)
    # 测试预测结果
    # clf.fit(trainMat, Classlabels)
    y_pred = clf.predict(testDoc)
    return y_pred


def getAccuracy(y_pred=np.array([]), real_label=np.array([])):
    ac = 0
    if y_pred.__len__() == real_label.__len__():
        for i in range(y_pred.__len__()):
            if y_pred[i] == real_label[i]:
                ac += 1
        return ac / (y_pred.__len__() + 0.0)
    else:
        print('Variable length is not valid')
        return None


# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

def single_main():
    for i in header:
        print('当前 特征{}'.format(i))
        train_mat, train_class_label = loading_train_data_in_single_vector(i)
        test_mat, test_class_label = loading_test_data_in_single_vector(i)
        ac = getAccuracy(MyGaussianNB(trainMat=train_mat, Classlabels=train_class_label, testDoc=test_mat),
                         test_class_label)
        print("最大似然估计 By Sk-Learn {}".format(ac))
        ac = getAccuracy(MyMultinomialNB(trainMat=train_mat, Classlabels=train_class_label, testDoc=test_mat),
                         test_class_label)
        print("贝叶斯估计法 By Sk-Learn {}".format(ac))


def mutiple_main():
    print('考虑 三种特征的情况')
    train_mat, train_class_label = loading_train_data()
    test_mat, test_class_label = loading_test_data()
    ac = getAccuracy(MyGaussianNB(trainMat=train_mat, Classlabels=train_class_label, testDoc=test_mat),
                     test_class_label)
    print("最大似然估计 By Sk-Learn {}".format(ac))
    ac = getAccuracy(MyMultinomialNB(trainMat=train_mat, Classlabels=train_class_label, testDoc=test_mat),
                     test_class_label)
    print("贝叶斯估计法 By Sk-Learn {}".format(ac))


def ROCCurve_new():
    print("ROC 曲线 相关参数与绘制 - 三种特征")
    train_mat, train_class_label = loading_train_data()
    test_mat, test_class_label = loading_test_data()
    X = np.array(train_mat)
    Y = np.array(train_class_label)
    # 高斯分布
    clf = naive_bayes.GaussianNB()
    clf.set_params(priors=[0.6776119402985075, 0.32238805970149254])
    # 设置先验概率
    clf.fit(X, Y)
    # print(clf.predict_proba(np.array([[171, 68, 38]])))
    n = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    sore_list = []
    for i in test_mat:
        sore_list.append(clf.predict_proba(np.array([i]))[0][1])
        label = clf.predict(np.array([i]))
        if (label == 1 and test_class_label[n] == 1):
            TP += 1
        elif (label == 0 and test_class_label[n] == 0):
            TN += 1
        elif (label == 1 and test_class_label[n] == 0):
            FP += 1
        else:
            FN += 1
        n += 1
    ccuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print("F1：", F1)
    print("准确率：", ccuracy)
    print("查准率：", precision)
    print("召回率：", recall)
    # data = np.append([sore_list], [test_class_label], axis=0)
    # data = data[np.lexsort(-data[:, ::-1].T)]
    #
    # sore = data[:, 0]
    # label = data[:, 1]
    # print(sore)
    # print(label)
    sore = sore_list
    label = test_class_label
    false_positive_rate, true_positive_rate, thresholds = roc_curve(label, sore)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('ROC by sk-learn')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f  Accuracy= %0.2f' % (roc_auc, ccuracy*100))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # print(test_mat)


if __name__ == '__main__':
    single_main()
    mutiple_main()
    multi_vector.main_output()
    print("ROC 曲线部分-----------------")
    ROCCurve_new()
    multi_vector.ROCCurve_me()
    single_vector.main_proc()
