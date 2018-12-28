import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
import itertools

all_classes = [1.0, 2.0, 3.0, 4.0]
all_classes_iter = itertools.combinations(all_classes, r=2)


def loading_train_data():
    result_mat = pd.read_csv('../../data_loader/gener/ellipse_data.txt',
                             names=['x', 'y', 'label'], sep=',')
    class_label = result_mat.pop('label')
    return np.array(result_mat.values), np.array(class_label.values)


def loading_variables():
    b_dict = {}
    ws_dict = {}
    for i in all_classes_iter:
        b, alphas, ws = pickle.load(open('./resultdata_label{}-{}.txt'.format(i[0], i[1]), 'rb'))
        b_dict["{}-{}".format(i[0], i[1])] = b
        ws_dict["{}-{}".format(i[0], i[1])] = ws
    return b_dict, ws_dict


def select_best_label(label, predict):
    return int(label[0]) if predict > 0 else int(label[-3])


def cal():
    result_mat, class_label = loading_train_data()
    b_dict, ws_dict = loading_variables()
    X_train, X_test, y_train, y_test = train_test_split(result_mat, class_label, test_size=0.33)
    ac = 0
    result_list = []
    print(b_dict)
    print(ws_dict)
    for single_data_index in range(X_test.__len__()):
        sore_list = [0] * 4
        for compete_label_pair in itertools.combinations(all_classes, r=2):
            variable_dict_index = "{}-{}".format(compete_label_pair[0], compete_label_pair[1])
            sore_list[select_best_label(variable_dict_index,
                                        (X_test[single_data_index] * np.mat(ws_dict[variable_dict_index]) + b_dict[
                                            variable_dict_index]).tolist()[
                                            0][0]) - 1] += 1
            # U Needn't know what i am doing !!!
            best_label = sore_list.index(np.array(sore_list).max()) + 1
        if y_test[single_data_index] == best_label:
            ac += 1
        result_list.append(best_label)
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
    ax.set_title('Four Type Ellipse with OVO_SVMs with Ac %.2f' % (file_name))
    cax = ax.matshow(mat, interpolation='nearest')
    print('[*] Accurate {}'.format(file_name))
    fig.colorbar(cax)
    plt.savefig('mat.png')
    plt.show()


if __name__ == '__main__':
    y_test, result_list, ac = cal()
    y_test = np.array(y_test, dtype='int32')
    result_list = np.array(result_list, dtype='int32')
    mat = to_matrix(y_test, result_list)
    mat_visible(mat, file_name=ac)
