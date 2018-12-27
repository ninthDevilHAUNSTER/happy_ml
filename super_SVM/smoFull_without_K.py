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


class optStruct(object):
    '''
    更加优质的数据结构
    设置了缓存误差，建立一个数据结构来保存重要的值
    '''

    def __init__(self, dataMatIn, classLabels, C, toler, kTup=None):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))  # 创建一个alpha向量，并初始化为0向量。实际上它也被叫做 $\w$
        self.b = 0
        # eCache 第一列是有效的标志位 第二列是实际的E值
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag

        # self.K = mat(zeros((self.m, self.m)))
        # for i in range(self.m):
        #     self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    '''
    计算e值并返回
    :param oS:
    :param k:
    :return:
    '''
    # fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    '''
    不再是随机选择 j
    而是把它给算出来。函数的误差值和 Ei , i 有关系
    :param i:
    :param oS:
    :param Ei:
    :return:
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    # 构建一个非 0 表 包含输入列表为目录的列表值，返回非零E值对应的alpha值
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        # 在 非0表 中选择一个长度最大的 j 值并算出 Ej 作为返回值
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # 别算了，浪费时间
            Ek = calcEk(oS, k)
            #  这里的 deltaE 就是 Ei 和 Ek 的差值
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    # 如果 oS.eCache[k] 为 1 那么,表示这个地方的值已经被修改,其值为 Ek
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    '''
    内循环代码，返回0，1.
    :param i: 当前下标
    :param oS: 数据结构
    :return: 0 or 1
            0 represents un modified
            1 represents modified
    '''
    # 首先 Ei 是可以被算出来的
    Ei = calcEk(oS, i)
    # ----------- 如果这个 i 可以被优化的话 -----------
    # 也就意味着 C ≥ a ≥ 0  &  -容错率 ≤ SUM(a_i·label^(i)) ≤ 容错率 。 toler 当然，这里可能是负的，可能是正的。所以考虑两种情况
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # ----------- 找出一个可以被优化的 k -----------
        j, Ej = selectJ(i, oS, Ei)
        # 计算两个向量之间的距离 计为 Ej
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # ----------- 保证 alpha 在 0 和 C 之间 -----------
        # 在迭代之前，先保证 随机选择的 alphas值是有效的
        # 确定一个最大，最小值 对 i 进行修改，改为和j方向相同，但是方向相反。
        # 用于将 alpha[j] 调整到 0-C 之间，如果L = H 不做任何改变
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 求出 最优修改量
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T \
              - oS.X[j, :] * oS.X[j, :].T
        # eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        # 在 eta 的帮助下 更新 alphas[j]
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 更新迭代误差
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # 如果改变过于轻微，则退出去
            print("j not moving enough")
            return 0
        # 改变完 alphas[j]后，对 i 也要做出改变，方向与 j 相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)
        # ----------- 计算常数项 -----------
        # b1 通过 Ei 来计算 b2 通过 Ej 来计算
        # b1 = oS.b - Ei
        # - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i]
        # - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        # b2 = oS.b - Ej
        # - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j]
        # - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        b1 = oS.b - Ei \
             - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T \
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej \
             - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T \
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 看下 b 取多少比较合适
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):  # full Platt SMO
    '''
    外循环代码
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :return:
    '''
    # oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历循环所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 当遍历过一次后，将entireSet设为false.这样就可以遍历所有边界值，来减少计算成本
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('../data_loader/testSet.txt')
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas > 0])
    print(calcWs(alphas, dataMat, labelMat))
