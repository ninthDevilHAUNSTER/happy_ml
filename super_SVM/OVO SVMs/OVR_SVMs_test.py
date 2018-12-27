import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize


def loading_train_data():
    result_mat = pd.read_csv('../../data_loader/gener/ellipse_data_easy_to_splict.txt',
                             names=['x', 'y', 'label'], sep=',')
    class_label = result_mat.pop('label')
    return np.array(result_mat.values), np.array(class_label.values)


def loading_variables():
    b_dict = {}
    ws_dict = {}
    for i in range(1, 5):
        b, alphas, ws = pickle.load(open('./resultdata_label{}.txt'.format(i), 'rb'))
        b_dict[i] = b
        ws_dict[i] = ws
    return b_dict, ws_dict


def select_best_label(result__):
    best_label = 0
    for i in result__:
        if i > best_label:
            best_label = i
    return result__.index(best_label) + 1


def cal():
    result_mat, class_label = loading_train_data()
    b_dict, ws_dict = loading_variables()
    X_train, X_test, y_train, y_test = train_test_split(result_mat, class_label, test_size=0.33)
    ac = 0
    result_list = []
    for i in range(X_test.__len__()):
        single_data = X_test[i]
        single_label = y_test[i]
        result__ = []
        for j in range(1, 5):
            result__.append((single_data * np.mat(ws_dict[j]) + b_dict[j]).tolist()[0][0])
        result_list.append(select_best_label(result__))
        if single_label == select_best_label(result__):
            ac += 1.0

    return y_test, result_list, ac / X_test.__len__()


def to_matrix(actual, predict):
    # Result_column
    confusion_matrix = np.zeros((
        max(actual) + 1, max(predict) + 1))
    for i in range(len(actual)):
        confusion_matrix[actual[i]][predict[i]] += 1
    return confusion_matrix


def mat_visible(mat, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Four Type Ellipse with OVR SVMs with Ac %.2f' % (file_name))
    cax = ax.matshow(mat, interpolation='nearest')
    fig.colorbar(cax)
    plt.show()


if __name__ == '__main__':
    y_test, result_list, ac = cal()
    y_test = np.array(y_test, dtype='int32')
    result_list = np.array(result_list, dtype='int32')
    mat = to_matrix(y_test, result_list)
    mat_visible(mat, file_name=ac)
