from data_loader.data_loader import ex0_loading_data
from numpy import *


def standRegres(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=0.005):  # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 线性回归
    xArr, yArr = ex0_loading_data()
    yHat = lwlrTest(xArr, xArr, yArr.T)
    print(yHat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).flatten().A[0], s=2, c='red')
    plt.show()
# ws = standRegres(xArr, yArr)
# print(ws)
# 局部线性回归
