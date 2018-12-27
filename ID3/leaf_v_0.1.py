import os, sys

# import g
sys.path.append('../MLiA_SourceCode/Ch03')
from pprint import pprint
import numpy as np
import pandas as pd
from pprint import pprint
from trees import createTree
# from trees import createTree
from treePlotter import createPlot

header = 'Class, Specimen Number, Eccentricity, Aspect Ratio, Elongation, Solidity, Stochastic Convexity, Isoperimetric\
          Factor, Maximal Indentation Depth, Lobedness, Average Intensity, Average Contrast, Smoothness, Third moment,\
          Uniformity, Entropy'.replace(' ', '').split(',')
# 类，样本数，偏心率，纵横比，伸长率，压缩，随机凸性，等周期
#            因子，最大压痕深度，Lobedness，平均强度，平均对比度，平滑度，第三时刻，
#            均匀性，熵
print(header)

for_train_data = pd.read_csv('./for train.csv', names=header, sep=',')
# print(for_train_data.values)
for_train_data.drop('Specimen Number'.replace(' ', '').split(','), axis=1, inplace=True)
class_column = for_train_data.pop('Class')
for_train_data.insert(header.__len__() - 2, header[0], class_column)
print(for_train_data)
# for_train_data.values.to
step_array2 = np.array(
    [0.0088133, 0.158254, 0.0083768, 0.0048427, 0.0060351, 0.00779784, 0.001961435, 0.072047357, 0.001440281,
     0.00231385, 0.000644077, 0.000295566, 2.92704E-05, 0.0248303, 1]
)
step_array = np.array([
    0.0440665,
    0.79127,
    0.041884,
    0.0242135,
    0.0301755,
    0.0389892,
    0.009807175,
    0.360236785,
    0.007201405,
    0.01156925,
    0.003220385,
    0.001477829,
    0.000146352,
    0.1241515,
    1])
lisanhua_data = for_train_data.values / step_array
lisanhua_data = np.array(lisanhua_data, dtype='int32')
print(lisanhua_data)

header.remove("SpecimenNumber")
header.append(header.pop(0))
# print(lisanhua_data)
lensesTree = createTree(lisanhua_data.tolist(), header)
print(lensesTree)
createPlot(lensesTree)


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

# storeTree(lensesTree, 'leafTree.txt')
