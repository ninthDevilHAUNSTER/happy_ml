from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from smoFull_within_K import calcWs, smoP
from data_loader.data_loader import boy_and_girl_loading_train_data_in_single_vector, \
    boy_and_girl_loading_test_data_in_single_vector
from sklearn.metrics import roc_curve, auc


def loading_train_data_with_shoe_size_and_height():
    X, Y = boy_and_girl_loading_train_data_in_single_vector('shoe size')
    X1, Y1 = boy_and_girl_loading_train_data_in_single_vector('height')
    dataMat = np.column_stack((X, X1))
    labelMat = np.array([1] * Y.tolist().count(1) + [-1] * Y.tolist().count(0))
    return dataMat, labelMat


def loading_test_data_with_shoe_size_and_height():
    X, Y = boy_and_girl_loading_test_data_in_single_vector('shoe size')
    X1, Y1 = boy_and_girl_loading_test_data_in_single_vector('height')
    dataMat = np.column_stack((X, X1))
    labelMat = np.array([1] * Y.tolist().count(1) + [-1] * Y.tolist().count(0))
    return dataMat, labelMat


def predict_value(ws, b, data):
    return data * np.mat(ws) + b


def test_in_smoP():
    dataMat, labelMat = loading_train_data_with_shoe_size_and_height()
    b, alphas = smoP(dataMat, labelMat, 0.1, 0.00001, 100000, ('lin', 0.6))
    ws = calcWs(alphas, dataMat, labelMat)
    dataMat, labelMat = loading_test_data_with_shoe_size_and_height()
    score_list = []
    ac = 0
    for i in range(dataMat.__len__()):
        score_list.append(predict_value(ws, b, dataMat[i]).tolist()[0])
        if predict(ws, b, dataMat[i]) == labelMat[i]:
            ac += 1
    print("准确率 {}".format(ac / dataMat.__len__()))

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
    for i in range(dataMat.__len__()):
        xPt = float(dataMat[i][0])
        yPt = float(dataMat[i][1])

        if x_min > xPt:
            x_min = xPt
        if x_max < xPt:
            x_max = xPt
        if y_min > yPt:
            y_min = yPt
        if y_max < yPt:
            y_max = yPt
        label = float(labelMat[i])
        predict_label = predict(ws, b, [xPt, yPt])
        # print(predict_label == label)
        if predict_label == label:
            if predict_label == 1:
                xcord_1_True.append(xPt)
                ycord_1_True.append(yPt)
            else:
                xcord_0_True.append(xPt)
                ycord_0_True.append(yPt)
        else:
            if predict_label == 1:
                xcord_1_False.append(xPt)
                ycord_1_False.append(yPt)
            else:
                xcord_0_False.append(xPt)
                ycord_0_False.append(yPt)
    plt.xlabel('height')
    plt.ylabel('shoe size')
    plt.title('SVC with linear kernel')

    plt.scatter(xcord_1_True, ycord_1_True, s=75, c='darkblue', alpha=0.5, label='1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(xcord_0_True, ycord_0_True, s=75, c='darkred', alpha=0.5, label='-1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(xcord_1_False, ycord_1_False, s=75, c='yellow', alpha=0.5,
                label='1_False')  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(xcord_0_False, ycord_0_False, s=75, c='green', alpha=0.5,
                label='-1_False')  # s为size，按每个点的坐标绘制，alpha为透明度
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter(x, y, s=120, c='', alpha=0.5, linewidth=3, edgecolor='purple')

    plt.xlim(34.0, 46.994117647060364)
    plt.ylim(145, 188)
    plt.legend()
    plt.show()

    ROCCurve_me(ccuracy=ac / dataMat.__len__(), result_list=labelMat, sore_list=score_list)


def predict(ws, b, data):
    return 1.0 if data * np.mat(ws) + b > 0 else -1.0


def ROCCurve_me(result_list, sore_list, ccuracy):
    # sore = np.array(sore_list, dtype='float')
    print(ccuracy)
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
    test_in_smoP()
