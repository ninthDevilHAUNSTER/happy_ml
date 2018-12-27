import sys

sys.path.append('D:\python_box\happy_ml')

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from smoFull_within_K import calcWs, smoP
from data_loader.data_loader import boy_and_girl_loading_test_data, boy_and_girl_loading_train_data
from mpl_toolkits.mplot3d import Axes3D


def loading_train_data_within_all_vector():
    dataMat, Y = boy_and_girl_loading_train_data()
    labelMat = np.array([1] * Y.tolist().count(1) + [-1] * Y.tolist().count(0))
    return dataMat, labelMat


def loading_test_data_within_all_vector():
    dataMat, Y = boy_and_girl_loading_test_data()
    labelMat = np.array([1] * Y.tolist().count(1) + [-1] * Y.tolist().count(0))
    return dataMat, labelMat


def smoP_main_code():
    dataMat, labelMat = loading_train_data_within_all_vector()
    b, alphas = smoP(dataMat, labelMat, 1, 0.001, 10000, ('rbf', 2))
    # 哭了 rbf 核只能到 68
    ws = calcWs(alphas, dataMat, labelMat)
    ac = 0
    for i in range(dataMat.__len__()):
        if predict(ws, b, dataMat[i]) == labelMat[i]:
            ac += 1
    print("准确率 {}".format(ac / dataMat.__len__()))
    return ws, b


def test_in_smoP():
    dataMat, labelMat = loading_train_data_within_all_vector()

    ws, b = [[0.22185989], [0.24766555], [0.3368476]], [[-66.98664169]]
    # 这个参数到 80% 极限了....
    ac = 0
    for i in range(dataMat.__len__()):
        if predict(ws, b, dataMat[i]) == labelMat[i]:
            ac += 1
    print("准确率 {}".format(ac / dataMat.__len__()))

    xcord_1_True = []
    xcord_0_True = []
    xcord_1_False = []
    xcord_0_False = []

    ycord_1_True = []
    ycord_0_True = []
    ycord_1_False = []
    ycord_0_False = []

    zcord_1_True = []
    zcord_0_True = []
    zcord_1_False = []
    zcord_0_False = []

    for i in range(dataMat.__len__()):
        xPt = float(dataMat[i][0])
        yPt = float(dataMat[i][1])
        zPt = float(dataMat[i][2])
        label = float(labelMat[i])
        predict_label = predict(ws, b, [xPt, yPt, zPt])
        # print(predict_label == label)
        if predict_label == label:
            if predict_label == 1:
                xcord_1_True.append(xPt)
                ycord_1_True.append(yPt)
                zcord_1_True.append(zPt)
            else:
                xcord_0_True.append(xPt)
                ycord_0_True.append(yPt)
                zcord_0_True.append(zPt)
        else:
            if predict_label == 1:
                xcord_1_False.append(xPt)
                ycord_1_False.append(yPt)
                zcord_1_False.append(zPt)
            else:
                xcord_0_False.append(xPt)
                ycord_0_False.append(yPt)
                zcord_0_False.append(zPt)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('height')
    ax.set_ylabel('wight')
    ax.set_zlabel('shoe size')
    ax.scatter(xcord_1_True, ycord_1_True, zcord_1_True, s=75, c='darkblue', alpha=0.5,
               label='1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    ax.scatter(xcord_0_True, ycord_0_True, zcord_0_True, s=75, c='darkred', alpha=0.5,
               label='-1_True')  # s为size，按每个点的坐标绘制，alpha为透明度
    ax.scatter(xcord_1_False, ycord_1_False, zcord_1_False, s=75, c='yellow', alpha=0.5,
               label='1_False')  # s为size，按每个点的坐标绘制，alpha为透明度
    ax.scatter(xcord_0_False, ycord_0_False, zcord_0_False, s=75, c='green', alpha=0.5,
               label='-1_False')  # s为size，按每个点的坐标绘制，alpha为透明度
    ax.set_ylim(40, 80)
    ax.set_zlim(34.0, 47)
    ax.set_xlim(155, 188)
    ax.legend()
    ax.view_init(elev=10, azim=235)
    plt.savefig('fig.png', bbox_inches='tight')
    plt.show()


def predict(ws, b, data):
    return 1.0 if data * np.mat(ws) + b > 0 else -1.0


if __name__ == '__main__':
    test_in_smoP()
