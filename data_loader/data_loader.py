import pandas as pd
import numpy as np
import sys

# global variables
boy_and_girl_header = ['height', 'weight', 'shoe size']


def ex0_loading_data():
    ex0_data = pd.read_csv('D:/python_box/happy_ml/data_loader/ex0.txt', names=['id', 'x', 'y'], sep='\t')
    # ex0_data.pop('id')
    ex0_data.pop('y')

    class_lables = pd.read_csv('D:/python_box/happy_ml/data_loader/ex0.txt', names=['id', 'x', 'y'], sep='\t')
    class_lables.pop('id')
    class_lables.pop('x')

    return np.array(ex0_data.values), np.array(class_lables.values)


def boy_and_girl_loading_train_data_in_single_vector(vector_name):
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('D:/python_box/happy_ml/data_loader/boy.txt', names=['height', 'weight', 'shoe size'],
                           sep='\t')
    girl_data = pd.read_csv('D:/python_box/happy_ml/data_loader/girl.txt', names=['height', 'weight', 'shoe size'],
                            sep='\t')

    for i in boy_and_girl_header:
        if vector_name != i:
            boy_data.pop(i)
            girl_data.pop(i)

    girl_data = np.array(girl_data.values)
    boy_data = np.array(boy_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_lables = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_lables


def boy_and_girl_loading_test_data_in_single_vector(vector_name):
    '''
    'height', 'weight', 'shoe size','class'
    :return: np array
    '''
    boy_data = pd.read_csv('D:/python_box/happy_ml/data_loader/boynew.txt', names=['height', 'weight', 'shoe size'],
                           sep=' ')
    girl_data = pd.read_csv('D:/python_box/happy_ml/data_loader/girlnew.txt', names=['height', 'weight', 'shoe size'],
                            sep=' ')

    for i in boy_and_girl_header:
        if vector_name != i:
            boy_data.pop(i)
            girl_data.pop(i)

    girl_data = np.array(girl_data.values)
    boy_data = np.array(boy_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_label = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_label


def boy_and_girl_loading_train_data():
    '''
    'height', 'weight', 'shoe size' | 'class'
    :return: result_mat, class_lables
    '''
    boy_data = pd.read_csv('D:/python_box/happy_ml/data_loader/boy.txt', names=['height', 'weight', 'shoe size'],
                           sep='\t')
    # boy_data.insert(boy_and_girl_header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('D:/python_box/happy_ml/data_loader/girl.txt', names=['height', 'weight', 'shoe size'],
                            sep='\t')
    # girl_data.insert(boy_and_girl_header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_lables = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_lables


def boy_and_girl_loading_test_data():
    '''
    'height', 'weight', 'shoe size' | 'class'
    :return: result_mat, class_lables
    '''
    boy_data = pd.read_csv('D:/python_box/happy_ml/data_loader/boynew.txt', names=['height', 'weight', 'shoe size'],
                           sep=' ')
    # boy_data.insert(boy_and_girl_header.__len__(), 'class', [1] * boy_data.values.__len__())
    boy_data = np.array(boy_data.values)
    girl_data = pd.read_csv('D:/python_box/happy_ml/data_loader/girlnew.txt', names=['height', 'weight', 'shoe size'],
                            sep=' ')
    # girl_data.insert(boy_and_girl_header.__len__(), 'class', [0] * girl_data.values.__len__())
    girl_data = np.array(girl_data.values)
    result_mat = np.vstack((boy_data, girl_data))
    class_label = np.array([1] * boy_data.__len__() + [0] * girl_data.__len__())
    return result_mat, class_label

print(boy_and_girl_loading_train_data_in_single_vector('height'))
