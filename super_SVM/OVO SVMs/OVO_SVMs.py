import os
import sys

sys.path.append(os.path.abspath('../../'))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from super_SVM.base.smoFull_within_K import calcWs, smoP


def predict(ws, b, data):
    return 1.0 if data * np.mat(ws) + b > 0 else -1.0


def predict_value(ws, b, data):
    return data * np.mat(ws) + b


def loading_train_data():
    result_mat = pd.read_csv('../..\data_loader\gener\ellipse_data.txt',
                             names=['x', 'y', 'label'], sep=',')
    class_label = result_mat.pop('label')
    return np.array(result_mat.values), np.array(class_label.values)


def ovo_svm_main(result_mat=np.array([]), class_label=np.array([]),
                 positive_label='1'):
    b, alphas = smoP(result_mat, class_label, C=10, toler=0.001, maxIter=200, kTup=('lin', 1.3))
    ws = calcWs(alphas, result_mat, class_label)
    return b, ws


def main():
    result_mat, class_label = loading_train_data()
    X_train, X_test, y_train, y_test = train_test_split(result_mat, class_label, test_size=0.33)
    class_label_list = [1.0, 2.0, 3.0, 4.0]
    super_variables = []
    for i in class_label_list:
        two_part_class_label = y_train.copy()
        for j in range(two_part_class_label.__len__()):
            if two_part_class_label[j] == i:
                two_part_class_label[j] = 1
            else:
                two_part_class_label[j] = -1
            b, ws = ovo_svm_main(X_train, two_part_class_label)
            super_variables.append([b, ws])
        print("[*] one part done")
    print(super_variables)


if __name__ == '__main__':
    main()    # X_train, X_test, y_train, y_test = train_test_split(result_mat, class_label, test_size=0.33)
