import sys, os
import random
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    SMO 函数的伪代码如下

    创建一个alpha向量，并初始化为0向量
    设定迭代次数
        对每个数据集中的数据向量
            如果该数据向量可以被优化
                随机选择另外一个数据向量
                优化这两个数据向量
                如果这两个向量都不能优化，推出内循环
        如果所有向量都没有优化，增加一次迭代次数，继续下一次循环

    :param dataMatIn: 输入的数据集
    :param classLabels: 输入的分类集
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 最大递归次数
    :return:
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))  # 创建一个alpha向量，并初始化为0向量。实际上它也被叫做 $\w$
    iter = 0
    # 外循环
    while (iter < maxIter):
        alphaPairsChanged = 0  # 该变量记录alpha是否被优化
        # 遍历整个数据集合
        for i in range(m):
            # 首先 fXi 是能够算出来的。（也就是预测的类别）
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 那么 Ei 就是 预测的类别和真实类别的距离
            Ei = fXi - float(labelMat[i])
            #  ----------- 如果该数据向量可以被优化 ----------------
            # 也就意味着 C ≥ a ≥ 0  &  -容错率 ≤ SUM(a_i·label^(i)) ≤ 容错率 。 toler 当然，这里可能是负的，可能是正的。所以考虑两种情况
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # ----------- 随机选择另外一个数据向量 -----------
                j = selectJrand(i, m)
                # ----------- 优化这两个数据向量 ----------- # 这就是内循环的过程
                # 计算两个向量之间的距离 计为 Ej
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()  # 更新alphas[i]的值。这样防止旧的东西被修改
                # ----------- 保证 alpha 在 0 和 C 之间 -----------
                # 在迭代之前，先保证 随机选择的 alphas值是有效的
                # 确定一个最大，最小值 对 i 进行修改，改为和j方向相同，但是方向相反。
                # 用于将 alpha[j] 调整到 0-C 之间，如果L = H 不做任何改变
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                #  这串这么长的玩意儿是求 最优修改量 在 \ 的帮助下，很好的知道这个东西是如何做到的
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                      - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    # 如果等于0的话会比较麻烦，但是现实中并不会这样子
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 在 eta 的帮助下，求得新的alphas[j]
                alphas[j] = clipAlpha(alphas[j], H, L)

                if (abs(alphas[j] - alphaJold) < 0.00001):
                    # 如果改变过于轻微，那么就退出去
                    print("j not moving enough")
                    continue
                # 改变完 alphas[j]后，对 i 也要做出改变，方向与 j 相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                # ----------- 计算常数项 -----------
                # b1 通过 Ei 来计算 b2 通过 Ej 来计算
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 看一下 b 取多少比较合适
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                # 确定改动次数 + 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    # 循环退出的条件是，迭代次数达到最多次。
    # 如果遍历整个数组，都没有更迭label标签，则迭代次数加一
    # 实际上会执行远多于迭代次数次的循环
    return b, alphas


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('../data_loader/testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 50)
    print(b)
    print(alphas[alphas>0])
