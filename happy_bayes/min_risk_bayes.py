import pandas as pd
import numpy as np
import math
import random
from pprint import pprint
from shaobaobaoer_math_lab import maximum_likelihood_estimation
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

# 训练样本 boys_new|girls_new
# 测试样本 boy|girl
# 将三种特征一起考虑，单类的概率相乘


# global variables
header = ['height', 'weight', 'shoe size']
b_rate = 227 / (108 + 227)
g_rate = 108 / (108 + 227)


# 1、处理数据：从csv文件中载入数据，然后划分为训练集和测试集。
def loading_test_data_return_two():
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
    被我写到标准库里去了
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
            # probabilities[classValue] *= maximum_likelihood_estimation(x, mean, stdev)
            probabilities[classValue] *= maximum_likelihood_estimation(x, mean, stdev) * (
                b_rate if classValue == 1.0 else g_rate)
        #     考虑先验概率
    # probabilities
    return probabilities


# 3、单一预测：使用数据集的特征生成单个预测。
def predict(summaries, inputVector):
    '''
    根据每个数据样本对于每个类的概率，返回应该属于的类。
    :param summaries: 之前生成的标准差和均值矩阵
    :param inputVector: 输入的单一数据样本
    :return:
    '''
    # print('#######################################')
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    # cal risk
    risk_1 = risk_0_1 * probabilities[0.0] + risk_1_1 * probabilities[1.0]
    risk_0 = risk_0_0 * probabilities[0.0] + risk_1_0 * probabilities[1.0]
    if risk_1 < risk_0:
        return 1.0, probabilities[1.0]
    else:
        return 0.0, probabilities[0.0]


# 4、多重预测：基于给定测试数据集和一个已提取特征的训练数据集生成预测。
def getPredictions(summaries, testSet):
    '''
    多重预测
    :param summaries:
    :param testSet:
    :return:
    '''
    predictions = []
    percentage = []
    for i in range(len(testSet)):
        result, m = predict(summaries, testSet[i])
        predictions.append(result)
        percentage.append(m)
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


def main_output():
    boy_data = loading_data()
    # print(boy_data)
    result = summarizeByClass(dataset=boy_data)
    print('最小贝叶斯分类器 {}'.format(result))
    boy_data = loading_test_data()
    result_list, percentage_list = getPredictions(result, boy_data)
    ac = getAccuracy(boy_data, result_list)
    print(percentage_list)
    print('贝叶斯估计法 By Me {}'.format(ac))


def score_list_convertion(score_list):
    sort_list = score_list.copy()
    sort_list.sort()
    min = sort_list[0]
    max = sort_list[-1]
    step = max - min / 100
    return np.array(score_list) / step


def ROCCurve_me():
    print("ROC曲线 by Me")
    boy_data = loading_data()
    result = summarizeByClass(dataset=boy_data)
    boy_data = loading_test_data()
    result_list, sore_list = getPredictions(result, boy_data)
    ccuracy = getAccuracy(boy_data, result_list)
    # sore_list = score_list_convertion(sore_list)
    sore = np.array(sore_list, dtype='float')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(result_list, sore)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC by Me')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f  Accuracy= %0.2f' % (roc_auc, ccuracy))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


'''
决策表

男=> 男： 0 男=>女： 7
女=> 男:  1 女=>女:  0

'''

if __name__ == '__main__':
    risk_1_1 = 0
    risk_1_0 = 7
    risk_0_0 = 0
    risk_0_1 = 7
    ROCCurve_me()
