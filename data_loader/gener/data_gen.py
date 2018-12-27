import numpy as np
from shaobaobaoer_math_lab import maximum_likelihood_estimation
from matplotlib import pyplot as plt
import os
import pandas as pd


def maximum_likelihood_e():
    x = np.arange(-5, 5, step=0.1)
    y = []
    for i in x:
        y.append(maximum_likelihood_estimation(i, 0, 1) * 10 * i / np.abs(i))
    y = np.array(y)
    return x, y


def ellipse(u, v, a, b):
    # u = 1.  # x-position of the center
    # v = 0.5  # y-position of the center
    # a = 2.  # radius on the x-axis
    # b = 1.5  # radius on the y-axis

    t = np.linspace(0, 2 * np.pi, 100)
    return u + a * np.cos(t), v + b * np.sin(t)


# ls_x, ls_y, color_map, label_map
def draw_plot(long_list):
    for i in range(long_list.__len__()):
        plt.plot(long_list[i][0], long_list[i][1], c=long_list[i][-2], label=long_list[i][-1])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("The Title")
    # plt.legend()
    plt.show()


def draw_and_renew_ellipse_data():
    long_list = []
    # x, y = maximum_likelihood_e()
    # long_list.append([x, y, 'r', 'max_liklyhood'])
    for u in range(1, 5):
        x, y = ellipse(3.5, 0, u * 0.09, 4)
        long_list.append([x, y, [1.0] * x.__len__(), 'b', 'ellipse x+ l'])
        x, y = ellipse(0, 3.3, 4, u * 0.1)
        long_list.append([x, y, [2.0] * x.__len__(), 'r', 'ellipse y+ -'])
        x, y = ellipse(-3.7, 0, u * 0.11, 4)
        long_list.append([x, y, [3.0] * x.__len__(), 'g', 'ellipse x- l'])
        x, y = ellipse(0, -3.2, 4, u * 0.1)
        long_list.append([x, y, [4.0] * x.__len__(), 'y', 'ellipse y- -'])
    # print(long_list[0][1])
    draw_plot(long_list)
    renew_data(long_list, file_name='ellipse_data')


def draw_and_renew_ellipse_data2():
    long_list1 = []
    long_list2 = []
    long_list3 = []
    long_list4 = []
    # x, y = maximum_likelihood_e()
    # long_list.append([x, y, 'r', 'max_liklyhood'])
    for u in range(1, 5):
        x, y = ellipse(4, 0, u * 0.1, 2.5)
        long_list1.append([x, y, [1.0] * x.__len__(), 'b', 'ellipse x+ l'])
        x, y = ellipse(0, 4, 2.5, u * 0.1)
        long_list2.append([x, y, [2.0] * x.__len__(), 'r', 'ellipse y+ -'])
        x, y = ellipse(-4, 0, u * 0.1, 2.5)
        long_list3.append([x, y, [3.0] * x.__len__(), 'g', 'ellipse x- l'])
        x, y = ellipse(0, -4, 2.5, u * 0.1)
        long_list4.append([x, y, [4.0] * x.__len__(), 'y', 'ellipse y- -'])
    # print(long_list[0][1])
    # draw_plot(long_list)
    renew_data(long_list1, file_name='ellipse_data_1')
    renew_data(long_list2, file_name='ellipse_data_2')
    renew_data(long_list3, file_name='ellipse_data_3')
    renew_data(long_list4, file_name='ellipse_data_4')


def renew_data(long_list, file_name):
    if os.path.exists(file_name.split('.')[0] + '.txt'):
        os.remove(file_name.split('.')[0] + '.txt')
        print('remove old data')

    for single_data in long_list:
        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'x': single_data[0], 'y': single_data[1], 'label': single_data[2]})
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(file_name.split('.')[0] + '.txt', index=False, sep=',', mode='a', header=None)


if __name__ == '__main__':
    draw_and_renew_ellipse_data()
