import os
import sys

sys.path.append(os.path.abspath('../../'))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from super_SVM.base.smoFull_within_K import calcWs, smoP
import pickle
import itertools


def predict(ws, b, data):
    return 1.0 if data * np.mat(ws) + b > 0 else -1.0


def predict_value(ws, b, data):
    return data * np.mat(ws) + b


def loading_train_data():
    result_mat = pd.read_csv('../../data_loader/gener/ellipse_data.txt',
                             names=['x', 'y', 'label'], sep=',')
    class_label = result_mat.pop('label')
    return np.array(result_mat.values), np.array(class_label.values)


def ovo_svm_main(result_mat=np.array([]), class_label=np.array([]),
                 positive_label="UK"):
    b, alphas = smoP(result_mat, class_label, C=1000, toler=0.0001, maxIter=200, kTup=('lin', 1.3))
    ws = calcWs(alphas, result_mat, class_label)
    pickle.dump([b, alphas, ws], open('./resultdata_label_{}.txt'.format(positive_label), 'wb'))
    return b, ws


def main():
    result_mat, class_label = loading_train_data()
    class_label_list = itertools.combinations([1.0, 2.0, 3.0, 4.0], r=2)
    # super_variables = []
    for i in class_label_list:
        two_part_class_label = []
        sub_result_mat = []
        label_active = "{}-{}".format(i[0], i[1])
        print("[+] Active Label {}".format(label_active))
        for j in range(class_label.__len__()):
            if class_label[j] == i[0]:
                two_part_class_label.append(1)
                sub_result_mat.append(result_mat[j])
            elif class_label[j] == i[1]:
                two_part_class_label.append(-1)
                sub_result_mat.append(result_mat[j])
            else:
                pass
        ovo_svm_main(np.array(sub_result_mat), np.array(two_part_class_label), positive_label=label_active)
        # super_variables.append([b, ws])
        print("[*] one part done")


if __name__ == '__main__':
    main()  # X_train, X_test, y_train, y_test = train_test_split(result_mat, class_label, test_size=0.33)
