import pickle
from pprint import pprint
import pandas as pd
import numpy as np

header = 'Class, Specimen Number, Eccentricity, Aspect Ratio, Elongation, Solidity, Stochastic Convexity, Isoperimetric\
          Factor, Maximal Indentation Depth, Lobedness, Average Intensity, Average Contrast, Smoothness, Third moment,\
          Uniformity, Entropy'.replace(' ', '').split(',')

Result_column_Real = [
    None, None, None, 1, None, 2, 14, 27, 26, 13, 9, 33, 24, None, 5, 5, 5, None, None, None, None, None, None, None,
    34, None, None, 24, 13, 13, None, 24, 10, 10, 36, 11, None, None, 6, None, 11, 12, None, 12, 22, 26, None, None, 13,
    None, 35, None, None, 15, 6, None, None, 36, None, 9, 1, 23, None, None, None, None, 14, 25, None, None, 26, None,
    15, 15, 24, None, 14, 15, None, None, None, None, None, 10, 10, 26, None, None, None, None, None, None, 1, 33, None,
    None, None, None, None, None, 14, None, None, None]
Result_column = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 9, 23, 23, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10,
                 10, 11, 11, 11, 11, 11, 11, 11, 5, 12, 7, 25, 24, 9, 30, 9, 14, 14, 7, 15, 15, 6, 22, 25, 28, 22, 23,
                 25, 23, 15, 23, 23, 24, 10, 25, 12, 26, 3, 23, 23, 3, 4, 28, 28, 28, 35, 25, 25, 7, 7, 9, 30, 3, 31,
                 31, 31, 31, 32, 32, 23, 30, 33, 33, 31, 31, 34, 12, 35, 35, 36, 11]

Result_column2 = [1, 1, 1, 1, 2, 2, 14, 27, 26, 13, 9, 33, 24, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 34, 9, 9, 24, 13, 13,
                  10, 24, 10, 10, 36, 11, 11, 11, 6, 11, 11, 12, 12, 12, 22, 26, 13, 13, 13, 14, 35, 14, 15, 15, 6, 22,
                  22, 36, 22, 9, 1, 23, 23, 24, 24, 24, 14, 25, 26, 26, 26, 26, 15, 15, 24, 28, 14, 15, 28, 29, 29, 29,
                  29, 10, 10, 26, 31, 31, 31, 31, 32, 32, 1, 33, 33, 33, 34, 34, 34, 35, 14, 35, 36, 36]

Result_column1 = []
for i in Result_column:
    if i is not None:
        Result_column1.append(i)
    else:
        Result_column1.append(-1)
    # Result_column.append(i if i is not None)


def to_csv(load_csv='for test.csv', output_csv='mativ.csv'):
    for_train_data = pd.read_csv(load_csv, names=header, sep=',')
    # print(for_train_data.values)
    # for_train_data.drop('Specimen Number'.replace(' ', '').split(','), axis=1, inplace=True)

    for_train_data.insert(header.__len__(), 'result', Result_column1)
    # print(for_train_data)
    for_train_data.to_csv(output_csv)


from matplotlib import pyplot as plt


def csv_to_img(input_csv='./OUTPUT.txt'):
    load_data = pd.read_csv(input_csv, sep='\t')
    load_data.drop('A', axis=1, inplace=True)
    load_data.drop('n', axis=1, inplace=True)
    mat = np.array(load_data.values, dtype='int32')
    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(mat, interpolation='nearest')
    fig.colorbar(cax)

    plt.show()


def mat_visible(mat, file_name):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(mat, interpolation='nearest')
    fig.colorbar(cax)

    plt.show()
    # (file_name + '.png')


def to_matrix(actual, predict):
    # Result_column
    confusion_matrix = np.zeros((
        max(actual) + 1, max(predict) + 1))
    for i in range(len(actual)):
        confusion_matrix[actual[i]][predict[i]] += 1
    return confusion_matrix


if __name__ == '__main__':
    # csv_to_img()
    load_csv = 'for test.csv'
    for_train_data = pd.read_csv(load_csv, names=header, sep=',')
    a = np.array(for_train_data.pop('Class'), dtype='int32')

    print('ID3 混淆矩阵结果')
    mat = to_matrix(a, Result_column)
    mat_visible(mat, file_name='ID3 混淆矩阵结果')
    print('C4.5 混淆矩阵结果')
    mat = to_matrix(a, Result_column2)
    mat_visible(mat, file_name='C4.5 混淆矩阵结果')
