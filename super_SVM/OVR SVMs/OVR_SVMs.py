import os
import sys

sys.path.append(os.path.abspath('../../'))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from super_SVM.base.smoFull_within_K import calcWs, smoP
import pickle


def predict(ws, b, data):
    return 1.0 if data * np.mat(ws) + b > 0 else -1.0


def predict_value(ws, b, data):
    return data * np.mat(ws) + b


def loading_train_data():
    result_mat = pd.read_csv('../../data_loader/gener/ellipse_data_easy_to_splict.txt',
                             names=['x', 'y', 'label'], sep=',')
    class_label = result_mat.pop('label')
    return np.array(result_mat.values), np.array(class_label.values)


def ovr_svm_main(result_mat=np.array([]), class_label=np.array([]),
                 positive_label=1):
    b, alphas = smoP(result_mat, class_label, C=1, toler=0.001, maxIter=20, kTup=('lin', 1.3))
    ws = calcWs(alphas, result_mat, class_label)
    dataframe = pd.DataFrame({'b': b, 'alphas': alphas, 'ws': ws}).reset_index()
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("result_{}.csv".format(positive_label), index=False, sep=',')
    return b, ws


def main():
    pass

if __name__ == '__main__':
    main()  # X_train, X_test, y_train, y_test = train_test_split(result_mat, class_label, test_size=0.33)
