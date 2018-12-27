Tree = {'height': {160: 36, 165: {'gender': {0: 37, 1: 41}}, 170: {'weight': {65: 41, 50: 41, 60: 42, 70: 42}},
                   175: {'weight': {80: 42, 65: 44, 70: 42}}, 180: {'weight': {80: 40, 65: 42, 75: 43, 70: 44}},
                   185: {'weight': {80: 46, 65: 41}}, 190: 43}}
Tree2 = {
    'height': {163: 1, 165: 0, 170: 1, 173: 1, 175: 1, 176: 1, 178: 1, 180: 1, 182: 1, 183: 1, 184: 1, 188: 1, 158: 0}}



Labels = ['gender', 'height', 'weight', 'size']
header_fast_list = {
    'height': 0, 'weight': 1, 'size': 2, 'gender': 3
}
import os, sys

import dicttoxml
from xml.dom.minidom import parseString

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
cs = pd.read_csv('./for test new.csv', names=Labels, sep=';')
class_column = cs.pop('gender')
cs.insert(Labels.__len__() - 1, Labels[0], class_column)

Labels.remove("size")
Labels.append(Labels.pop(0))


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
# import numpy as np

cs = np.array(cs, dtype='int32')
# cs = cs.tolist()


# print(cs)
def tree_to_xml(filename='tree', Tree=Tree):
    import json
    r = Tree
    r = json.dumps(r)
    Tree = json.loads(r)
    # 这么做是把int 的节点转成 str 的
    d = Tree
    xml = dicttoxml.dicttoxml(d, attr_type=False, root=False)
    print(xml)
    dom = parseString(xml)
    xml = dom.toprettyxml()
    with open(filename, 'w') as WXML:
        WXML.write(xml)


def curretion(result, l_st_data, current_tree, line):
    # print(l_st_data)
    for i in current_tree.findall("n{}".format(l_st_data)):
        if len(i):
            for child in i:
                if len(child) == 0:
                    return result
                l_st_data = line[header_fast_list[child.tag]]
                result = curretion(result, l_st_data, child, line)
        else:
            result = i.text
            print("Result :: {} Line[-1] ::{}".format(result,line[-1]))
            return result  # if result == line[-1] else False
    return result  # if result == line[-1] else False


def cal_accuracy(lisanhua_data, filename):
    cfrac_1_param = 0
    cfrac_2_param = lisanhua_data.__len__() + 0.0
    tree = ET.parse(filename)

    for line in lisanhua_data:
        line = line.tolist()
        l_st_data = line[header_fast_list[tree.getroot().tag]]
        result = curretion(line, l_st_data, tree, line)
        try:
            if int(line[-1]) == int(result):
                # print(cfrac_1_param)
                cfrac_1_param += 1
        except:
            pass
            print(result)
            print(line)
            # if int(line[-1]) == int(result[-1]):
            # cfrac_1_param += 1
    print(cfrac_1_param)
    print("[-] Header {}".format(Labels))
    print("[*] 准确率 {}".format(cfrac_1_param / cfrac_2_param))


import xml.etree.ElementTree as ET

if __name__ == '__main__':
    filename = 'student_tree_dict.xml'
    tree_to_xml(filename=filename, Tree=Tree2)
    cal_accuracy(cs, filename)
    # tree_to_xml(filename=filename,Tree=Tree_by_huang)
