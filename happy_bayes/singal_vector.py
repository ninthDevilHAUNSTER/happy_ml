import pandas as pd
import numpy as np
import math
import random
from pprint import pprint
from matplotlib import pyplot as plt
from sklearn import metrics

# 训练样本 boys_new|girls_new
# 测试样本 boy|girl
# 将三种特征一起考虑，单类的概率相乘
# 没有考虑先验概率和后验概率

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

# global variables
header = ['height', 'weight', 'shoe size']

header_color = {'height': 'b', 'weight': 'y', 'shoe size': 'r'}

b_rate = 227 / (108 + 227)
g_rate = 108 / (108 + 227)


# 1、处理数据：从csv文件中载入数据，然后划分为训练集和测试集。

# height(cm) | weight(kg) | shoe size(EUR)
def loading_data():
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('./dates/boy.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    boy_data.insert(header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('./dates/girl.txt', names=['height', 'weight', 'shoe size'], sep='\t')
    girl_data.insert(header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    # print('[*]\t男女数据集比例（不同先验概率）{}:{}'.format(boy_data.__len__(), girl_data.__len__()))
    return result_mat


# 2、提取数据特征：提取训练数据集的属性特征，以便我们计算概率并做出预测。

# 两类 3个数值属性

# 按类别划分数据
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# 计算均值
def mean(numbers):
    '''
    计算均值
    :param numbers:
    :return:
    '''
    return sum(numbers) / float(len(numbers))


# 计算标准差
def stdev(numbers):
    '''
    计算标准差
    :param numbers:
    :return:
    '''
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# 提取数据集特征
def summarize(dataset):
    '''
    提取数据集特征
    :param dataset: 数据集
    :return: 返回输入的每列的均值和标准差，为多对为二元组
    第一个是均值，第二个是标准差
    '''
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# 按类别提取属性特征
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def summarizeByClass(dataset):
    '''
    按类别提取属性特征
    :param dataset:
    :return:
    '''
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    '''
    最大似然估计（例示） 一元正态分布
    :param x:
    :param mean:
    :param stdev:
    :return:
    $$  A = e^{-\cfrac{(x_k-mean)^2}{2 stdev}} $$
    $$  return\; A* \cfrac{1}{\scrt{2*\pi*stdev}}$$
    '''
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    '''
    乘法合并概率，整个数据样本属于某个类的概率。
    :param summaries:
    :param inputVector:
    :return:
    '''
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def calculateClassProbabilities_in_one_vector(summaries, inputVector, single_vector):
    '''

    :param summaries:
    :param inputVector:
    :return:
    '''
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        i = header.index(single_vector)
        mean, stdev = classSummaries[i]
        x = inputVector[i]
        # probabilities[classValue] *= (x, mean, stdev)
        probabilities[classValue] *= calculateProbability(x, mean, stdev) * (
            b_rate if classValue == 1.0 else g_rate)
    return probabilities


# 3、单一预测：使用数据集的特征生成单个预测。
def predict(summaries, inputVector, single_vector):
    '''
    根据每个数据样本对于每个类的概率，返回应该属于的类。
    :param summaries: 之前生成的标准差和均值矩阵
    :param inputVector: 输入的单一数据样本
    :return:
    '''
    # print("predict{}{}".format(summaries, inputVector))
    probabilities = calculateClassProbabilities_in_one_vector(summaries, inputVector, single_vector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    # print('best label {}'.format(bestLabel))
    # print(str(bestLabel) + ',' + str(bestProb))
    return bestLabel, bestProb


# 4、多重预测：基于给定测试数据集和一个已提取特征的训练数据集生成预测。
def getPredictions(summaries, testSet):
    '''
    多重预测
    :param summaries:
    :param testSet:
    :return:
    '''
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getPredictions_in_one_vector(summaries, testSet, vector_index):
    '''
    多重预测
    :param summaries:
    :param testSet:
    :return:
    '''
    predictions = []
    percentage = []
    for i in range(len(testSet)):
        result, percent = predict(summaries, testSet[i], vector_index)
        predictions.append(result)
        percentage.append(percent)
    return predictions, percentage


# 5、评估精度：评估对于测试数据集的预测精度作为预测正确率。
def getAccuracy(testSet, predictions):
    '''

    :param testSet:
    :param predictions:
    :return:
    '''
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def loading_test_data():
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('./dates/boynew.txt', names=['height', 'weight', 'shoe size'], sep=' ')
    boy_data.insert(header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('./dates/girlnew.txt', names=['height', 'weight', 'shoe size'], sep=' ')
    girl_data.insert(header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    print('[*]\t男女数据集比例（不同先验概率）{}:{}'.format(boy_data.__len__(), girl_data.__len__()))

    return result_mat


import random


def magic_array(a, b):
    li = list(range(242))
    random.shuffle(li)
    c = []
    d = []
    for i in li:
        c.append(a[i])
        d.append(b[i])
    return c, d


def get_p(realzhi, yucezhi):
    FPR = []
    TPR = []

    for i in yucezhi:
        if realzhi == i:
            pass


def ROCCurve_me(tag_name='shoe size'):
    print("ROC曲线 by Me")
    boy_data = loading_data()
    result = summarizeByClass(dataset=boy_data)
    boy_data = loading_test_data()
    result_list, sore_list = getPredictions_in_one_vector(result, boy_data, tag_name)
    sore = np.array(sore_list, dtype='float')
    Accuracy = getAccuracy(boy_data, result_list)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(result_list, sore)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return false_positive_rate, true_positive_rate, tag_name, Accuracy, roc_auc


def main_proc():
    plt.title('ROC by me in single Vector')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1])
    plt.ylim([0, 1.05])
    plt.xlim([-0.1, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for i in header:
        false_positive_rate, true_positive_rate, tag_name, Accuracy, roc_auc = ROCCurve_me(tag_name=i)
        plt.plot(false_positive_rate, true_positive_rate, header_color[i],
                 # label='AUC = %0.2f' % (roc_auc))
                 label='AUC = %0.2f     %s Accuracy= %0.2f' % (roc_auc, tag_name, Accuracy))
    plt.legend()  # 显示图例
    plt.show()
