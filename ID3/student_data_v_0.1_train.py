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

Labels = ['gender', 'height', 'weight', 'size']
cs = pd.read_csv('./for test new.csv', names=Labels, sep=',')
class_column = cs.pop('gender')
cs.insert(Labels.__len__() - 1, Labels[0], class_column)
# num = len(cs)
# for i in range(num):
#     nu = cs.height[i] % 5
#     nu1 = cs.weight[i] % 5
#     if nu != 0:
#         if nu > (5 - nu):
#             cs.height[i] = (int)(cs.height[i] / 5 + 1) * 5
#         else:
#             cs.height[i] = (int)(cs.height[i] / 5) * 5
#     if nu1 != 0:
#         if nu1 > (5 - nu1):
#             cs.weight[i] = (int)(cs.weight[i] / 5 + 1) * 5
#         else:
#             cs.weight[i] = (int)(cs.weight[i] / 5) * 5
import numpy as np

cs = np.array(cs, dtype='int32')
cs = cs.tolist()
print(cs)

Labels.remove("size")
Labels.append(Labels.pop(0))

lensesTree = createTree(cs, Labels)
print(lensesTree)
createPlot(lensesTree)
