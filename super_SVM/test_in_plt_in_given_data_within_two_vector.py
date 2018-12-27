from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from smoFull_within_K import calcWs, smoP, loadDataSet
from sklearn.metrics import roc_curve, auc


def predict(ws, b, data):
    return 1.0 if data * np.mat(ws) + b > 0 else -1.0


def predict_value(ws, b, data):
    return data * np.mat(ws) + b


def test_in_plt():
    dataMat, labelMat = loadDataSet('../MLiA_SourceCode/Ch06/testSet.txt')
    b, alphas = smoP(dataMat, labelMat, 200, 0.001, 100000, ('lin', 1.3))
    ws = calcWs(alphas, dataMat, labelMat)

    ac = 0
    score_list = []
    for i in range(dataMat.__len__()):
        score_list.append(predict_value(ws, b, dataMat[i]).tolist()[0])
        if predict(ws, b, dataMat[i]) == labelMat[i]:
            ac += 1
    print("准确率 {}".format(ac / dataMat.__len__()))

    # ws = np.array([[0.75863406], [-0.17278613]])
    # b = np.array([[-3.54055206]])

    xcord_1_True = []
    xcord_0_True = []
    ycord_1_True = []
    ycord_0_True = []

    xcord_1_False = []
    xcord_0_False = []
    ycord_1_False = []
    ycord_0_False = []
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    fr = open('../MLiA_SourceCode/Ch06/testSet.txt')
    for line in fr.readlines():
        lineSplit = line.strip().split('\t')
        xPt = float(lineSplit[0])
        yPt = float(lineSplit[1])
        if x_min > xPt:
            x_min = xPt
        if x_max < xPt:
            x_max = xPt
        if y_min > yPt:
            y_min = yPt
        if y_max < yPt:
            y_max = yPt
        label = float(lineSplit[2])
        predict_label = predict(ws, b, [xPt, yPt])
        if predict_label == label:
            if predict_label == 1:
                xcord_1_True.append(xPt)
                ycord_1_True.append(yPt)
            else:
                xcord_0_True.append(xPt)
                ycord_0_True.append(xPt)
        else:
            if predict_label == 1:
                xcord_1_False.append(xPt)
                ycord_1_False.append(yPt)
            else:
                xcord_0_False.append(xPt)
                ycord_0_False.append(yPt)
    fr.close()

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVC with linear kernel')

    # n = 1024
    #
    # x = np.random.normal(0, 1, n)  # 平均值为0，方差为1，生成1024个数
    # y = np.random.normal(0, 1, n)
    # t = np.arctan2(x, y)  # for color value，对应cmap
    #
    plt.scatter(xcord_1_True, ycord_1_True, s=75, c='darkblue', alpha=0.5, label='1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(xcord_0_True, ycord_0_True, s=75, c='darkred', alpha=0.5, label='-1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(xcord_1_False, ycord_1_False, s=75, c='blue', alpha=0.5, label='1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(xcord_0_False, ycord_0_False, s=75, c='red', alpha=0.5, label='-1_True')  # s为size，按每个点的坐标绘制，alpha为透明度

    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter(x, y, s=120, c='', alpha=0.5, linewidth=3, edgecolor='purple')

    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()

    ROCCurve_me(ccuracy=ac,result_list=labelMat,sore_list=score_list)


def ROCCurve_me(result_list, sore_list, ccuracy):
    # sore = np.array(sore_list, dtype='float')
    print(result_list)
    print(sore_list)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(result_list, sore_list)
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

if __name__ == '__main__':
    test_in_plt()
