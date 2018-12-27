import pickle
from pprint import pprint
import pandas as pd
import numpy as np
Tree_by_huang2 = {'AspectRatio': {1: {'Lobedness': {0: {'Solidity': {36: {'StochasticConvexity': {32: 9, 33: 3}}, 37: {'Uniformity': {1: 3, 2: 9, 3: 9, 4: 9, 6: 9}}, 38: {'AverageContrast': {6: 23, 7: 4, 8: 23, 9: 3, 10: 9, 11: 30, 12: 3, 13: 3, 14: 30, 16: 30, 17: {'Eccentricity': {12: 30, 15: 3}}, 20: 10}}, 39: {'Thirdmoment': {0: 24, 1: 4, 2: {'Elongation': {8: 24, 4: 4, 6: 24, 7: 3}}, 3: {'StochasticConvexity': {32: 30, 33: 9}}, 4: 30, 6: 30, 11: 10, 15: 25}}, 40: {'MaximalIndentationDepth': {0: {'StochasticConvexity': {32: 33, 33: {'Eccentricity': {10: 26, 12: 26, 13: {'Elongation': {4: 27, 5: 26}}, 14: {'IsoperimetricFactor': {19: 4, 21: 13}}, 15: {'Uniformity': {1: 27, 3: 13, 4: 33, 6: 33}}, 16: {'Smoothness': {1: 1, 2: 1, 3: 26, 4: 13, 5: 1}}, 17: {'AverageIntensity': {1: 1, 4: 27, 6: 13}}}}}}, 1: {'Elongation': {2: 26, 3: 26, 4: 24, 6: 24, 7: {'Eccentricity': {13: 33, 14: 24, 15: 24}}, 8: 33}}}}, 41: 27, 31: 7}}, 1: {'Solidity': {35: 9, 36: 9, 37: {'Eccentricity': {7: 10, 10: 10, 12: 10, 14: 10, 16: 25, 18: 25}}, 38: 10, 39: 10}}, 2: {'Elongation': {10: {'Eccentricity': {10: 23, 12: 15}}, 11: 15, 12: 15, 17: 36, 18: 36}}, 3: {'Eccentricity': {9: 15, 3: 36, 12: 15, 6: 36}}, 4: {'Elongation': {19: 36, 18: 36, 11: 23}}, 5: 6, 6: {'Eccentricity': {11: {'Thirdmoment': {0: 11, 1: 6}}, 13: 36}}, 7: {'Eccentricity': {8: 11, 5: 11, 14: 6}}, 8: {'Uniformity': {0: {'IsoperimetricFactor': {4: 11, 7: 6}}, 2: 36}}, 9: 11, 10: 11, 11: {'Eccentricity': {10: 6, 6: 11}}}}, 2: {'Solidity': {33: {'Eccentricity': {19: 7, 20: 5}}, 34: {'IsoperimetricFactor': {8: 5, 9: 5, 10: 5, 12: 7}}, 35: 5, 36: 7, 37: 25, 38: {'Lobedness': {0: {'IsoperimetricFactor': {11: 12, 13: 22, 14: 12, 15: 7}}, 1: 25}}, 39: {'Smoothness': {0: 29, 5: {'Eccentricity': {19: 7, 20: 22}}, 6: 14}}, 40: {'Thirdmoment': {0: 1, 1: {'AverageIntensity': {2: 1, 3: {'Uniformity': {0: 27, 1: {'Eccentricity': {17: 1, 19: 32, 20: 32}}}}, 4: {'Eccentricity': {18: 27, 19: 2, 20: 2}}, 6: 27}}, 2: {'Elongation': {9: 33, 10: 27, 11: 26}}, 3: {'Eccentricity': {18: 33, 19: 32, 20: 32}}, 4: {'Eccentricity': {18: 33, 19: 32}}, 5: 14, 8: 28}}}}, 3: {'StochasticConvexity': {32: {'Entropy': {6: {'Solidity': {38: 22, 39: 12}}, 7: 22, 8: {'Eccentricity': {20: 22, 21: 14}}, 10: 12, 11: 12, 12: 28, 13: 12, 14: {'Eccentricity': {19: 12, 20: 14}}}}, 33: {'Smoothness': {1: 2, 2: 2, 4: 28, 5: 14, 6: 14, 7: {'Elongation': {13: 35, 14: 35, 15: 12}}, 8: 35, 9: 35, 10: {'Uniformity': {8: 35, 3: 14, 5: 28}}, 11: 28, 12: 14}}, 28: 5, 29: {'Solidity': {32: 22, 34: 5}}}}, 4: 28, 7: 8, 8: 8, 11: 31, 12: {'Smoothness': {1: 31, 2: 34}}, 13: {'Elongation': {21: 34, 22: 31}}, 14: 34, 15: 31, 19: 34, 21: 34}}
Tree = {'Elongation': {2: 26, 3: 26, 4: {'AverageContrast': {5: 27, 6: 24, 7: 24, 8: 24, 9: 26, 11: 4, 17: 30}},
                       5: {'Eccentricity': {4: 30, 7: 30, 10: 10, 11: 30, 12: 26, 13: 26}}, 6: {
        'IsoperimetricFactor': {10: 30, 12: 30, 13: 30, 14: 30, 15: 10, 17: 9, 18: 24,
                                19: {'Eccentricity': {12: 24, 14: 4}}, 20: 24, 21: 13}}, 7: {
        'Entropy': {2: 24, 3: 23, 4: {'Eccentricity': {13: 23, 15: 3}}, 6: 3,
                    7: {'Eccentricity': {16: 27, 12: 9, 14: 24}}, 8: {'Eccentricity': {11: 3, 13: 3, 15: 27}},
                    9: {'Eccentricity': {16: 1, 14: 4}}, 10: 9, 11: 10, 12: 33, 13: 10, 18: 33}}, 8: {
        'AverageContrast': {5: 1, 6: {'Solidity': {38: 23, 39: 24}}, 7: 1, 8: 26,
                            9: {'MaximalIndentationDepth': {0: 13, 1: 33, 3: 3}},
                            10: {'Eccentricity': {16: 13, 14: 24, 15: 13}}, 12: 9, 13: 33,
                            14: {'IsoperimetricFactor': {17: 3, 20: 13, 13: 9, 15: 3}}, 15: 9, 16: 10, 17: 3, 22: 10}},
                       9: {'AverageIntensity': {1: {'Eccentricity': {16: 29, 17: 1}}, 2: 33,
                                                3: {'Eccentricity': {17: 1, 18: 27}},
                                                4: {'Eccentricity': {16: 3, 17: 27, 18: 27}}, 5: 4, 6: 13, 7: 3, 8: 9,
                                                9: 9, 11: 10, 14: 25, 15: {'Eccentricity': {10: 10, 13: 9}}, 17: 25}},
                       10: {'AverageIntensity': {1: 1, 2: 1, 3: 23, 4: 27, 5: 15, 8: 33, 13: 25, 16: 10, 17: 25}}, 11: {
        'Entropy': {1: 29, 2: {'Eccentricity': {18: 29, 14: 23}}, 3: 23, 5: {'Eccentricity': {18: 26, 12: 23}}, 8: 7,
                    9: 7, 11: {'Eccentricity': {9: 15, 18: 33}}, 12: 25, 13: 27, 14: 25, 19: 15, 20: 15}}, 12: {
        'IsoperimetricFactor': {6: 15, 7: 7, 8: 6, 9: 7, 13: 22, 14: 12, 15: {'Solidity': {40: 32, 39: 29}}, 16: 28,
                                17: 14, 18: {'AverageIntensity': {4: 2, 5: 32, 6: 32, 7: 32}}}}, 13: {
        'AverageIntensity': {3: {'Eccentricity': {19: 7, 20: 32}}, 4: {'Eccentricity': {19: 5, 20: 2}}, 5: 32,
                             6: {'Eccentricity': {19: 7, 20: 22}}, 7: 7, 8: 14, 10: 35, 12: 15}}, 14: {
        'IsoperimetricFactor': {5: 11, 6: 6, 7: 6, 8: 5, 10: 12, 11: 12, 13: {'Solidity': {37: 22, 38: 22, 39: 14}},
                                14: {'AverageContrast': {8: 12, 12: 14, 14: 35, 16: 14, 17: 14}},
                                15: {'Smoothness': {8: 35, 10: 35, 11: 28}}, 16: 2}}, 15: {
        'IsoperimetricFactor': {3: 11, 4: 11, 8: 5, 9: 5, 10: 22, 11: 12, 12: {'Solidity': {38: 22, 39: 12}},
                                13: {'Solidity': {40: 35, 38: 12, 39: 22}},
                                14: {'AverageIntensity': {7: 14, 9: 28, 11: 35, 14: 28, 15: 28}},
                                15: {'Eccentricity': {20: 28, 21: 2}}}},
                       16: {'Solidity': {32: 22, 33: 5, 34: 5, 38: 14, 39: 28, 40: 35, 21: 11, 29: 6, 31: 6}},
                       17: {'Eccentricity': {6: 36, 8: 36, 10: 6, 13: 36, 21: 5}}, 18: 36,
                       19: {'Eccentricity': {8: 36, 10: 36, 22: 8}}, 20: {'Eccentricity': {10: 36, 22: 8}},
                       21: {'Entropy': {3: 31, 4: 34, 5: 34}}, 22: {'AspectRatio': {21: 34, 19: 34, 13: 31, 15: 31}}}}
Tree_by_huang = {'Thirdmoment': {0: {'Smoothness': {0: {'AverageContrast': {0: {'AverageIntensity': {2: 2, 3: {
    'Lobedness': {0: {'IsoperimetricFactor': {1: 1, 2: {'Entropy': {1: 1, 2: 2}}}},
                  1: {'IsoperimetricFactor': {0: 2, 1: 2, 2: 3, 4: 2}}}}, 4: {
    'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {2: 2, 3: 4}}, 11: {'StochasticConvexity': {2: 3, 5: 4}},
                                20: 3}}}}, 1: {'AverageIntensity': {
    4: {'Lobedness': {1: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {0: 3, 2: 2}}, 2: 3}}, 2: 5}}, 5: {
        'Lobedness': {1: {'MaximalIndentationDepth': {0: 2, 3: 3}},
                      2: {'MaximalIndentationDepth': {0: 4, 8: 4, 6: 5}}}},
    6: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {1: 5, 2: 4}}, 11: 5}}}}}}, 1: {'AverageContrast': {1: {
    'AverageIntensity': {5: 3, 6: {'Lobedness': {1: {'MaximalIndentationDepth': {0: 3, 4: 2}}, 2: {
        'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {0: 5, 1: 4, 2: 3}}, 8: 5, 6: 5}}, 3: 5}}}}, 2: {
    'AverageIntensity': {8: 6, 6: {'Lobedness': {2: {'IsoperimetricFactor': {0: 4, 2: 5}}, 3: 7}}, 7: {
        'Lobedness': {2: 5, 3: {'MaximalIndentationDepth': {0: {
            'IsoperimetricFactor': {0: {'StochasticConvexity': {16: 5, 20: {'AspectRatio': {8: 5, 9: 7}}, 21: 7}},
                                    2: {'StochasticConvexity': {4: 5, 7: 6}}, 4: 6}}, 6: 7}}}}}}}}, 2: {
    'AverageContrast': {2: {
        'AverageIntensity': {8: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {1: 4, 3: 8, 4: 4}}, 7: 6}},
                             7: {'Lobedness': {2: 3, 3: 5}}}}, 3: {
        'Lobedness': {3: {'MaximalIndentationDepth': {0: 6, 4: 5}}, 4: {'IsoperimetricFactor': {0: 7, 1: 5, 2: 7}}}}}},
                                                    3: {'AverageContrast': {2: 3, 3: {'AverageIntensity': {8: 4, 9: 5}},
                                                                            4: {'AverageIntensity': {9: 5, 10: {
                                                                                'MaximalIndentationDepth': {0: 9,
                                                                                                            2: 10}}}}}},
                                                    4: {'AverageIntensity': {9: 4, 10: 6}}, 5: 6,
                                                    7: {'AverageContrast': {6: 6, 7: 8}}, 8: 6, 11: 7}}, 1: {
    'Smoothness': {0: {'AverageIntensity': {5: 6, 6: {'MaximalIndentationDepth': {0: 5, 9: 6}}}}, 1: {
        'AverageContrast': {1: 5, 2: {'AverageIntensity': {
            8: {'IsoperimetricFactor': {0: {'StochasticConvexity': {16: 8, 20: 8, 21: 9, 22: 8}}, 1: 8, 2: 6}}, 7: {
                'Lobedness': {3: {'MaximalIndentationDepth': {
                    0: {'IsoperimetricFactor': {0: {'StochasticConvexity': {17: 6, 20: 5, 21: 7, 15: 5}}, 1: 4}},
                    10: 7}}, 4: 7}}}}, 3: 8}}, 2: {'AverageContrast': {3: 8, 4: {'Lobedness': {5: 8, 6: 10}}}}, 3: {
        'AverageContrast': {3: 8, 4: {'AverageIntensity': {10: {
            'Lobedness': {5: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {0: 6, 1: 7}}, 1: 8}},
                          6: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {0: 10, 4: 9}}, 3: 11}}}}, 11: 9}},
                            5: 9}}, 4: {
        'Lobedness': {6: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {2: 8, 3: 11}}, 6: 9}}, 7: 11}}, 5: {
        'AverageContrast': {5: 8, 6: {
            'Lobedness': {8: 10, 7: {'MaximalIndentationDepth': {0: {'IsoperimetricFactor': {2: 9, 3: 12}}, 9: 8}}}},
                            7: 11}}, 6: 9, 7: {'AverageContrast': {8: 9, 7: 8}}, 8: 8,
                   9: {'AverageContrast': {8: 7, 9: 8, 10: 12}}, 17: 11}}, 2: {
    'Smoothness': {0: {'AverageContrast': {1: 6, 2: 9}}, 1: {'AverageContrast': {1: 9, 2: 7, 3: 9}},
                   2: {'AverageIntensity': {9: 7, 10: {'Lobedness': {6: 10, 7: 11}}}}, 3: {
            'AverageContrast': {4: {'Lobedness': {5: 8, 6: 7, 7: 10}},
                                5: {'AverageIntensity': {11: {'Lobedness': {6: 7, 7: 9}}, 12: 10}}}},
                   4: {'AverageContrast': {6: {'MaximalIndentationDepth': {1: 13, 2: 14}}, 7: 12}},
                   5: {'AverageContrast': {8: 14, 6: {'AverageIntensity': {12: 9, 13: 11}}, 7: 13}},
                   6: {'AverageIntensity': {13: 8, 14: 10}}, 8: {'AverageContrast': {9: 10, 10: 11}}, 9: 12, 10: 12,
                   16: 13}}, 3: {'Smoothness': {1: 10, 2: {'Lobedness': {6: 9, 7: 11}}, 3: {
    'AverageContrast': {5: {'Lobedness': {8: 13, 7: {'IsoperimetricFactor': {2: 11, 3: 10}}}}, 6: 13}}, 4: 8,
                                                5: {'AverageContrast': {8: 14, 6: 10}},
                                                7: {'AverageContrast': {9: 18, 10: {'Solidity': {32: 14, 33: 13}}}},
                                                8: 15, 10: 13, 11: 12, 15: 12}}, 4: {
    'Smoothness': {1: 8, 4: {'AverageContrast': {6: 11, 7: 14}},
                   5: {'AverageContrast': {8: 13, 7: {'AverageIntensity': {13: 10, 14: 13}}}},
                   6: {'AverageIntensity': {14: 12, 15: 19}}, 7: 14, 8: {'AverageContrast': {9: 10, 11: 13, 12: 14}},
                   10: 12, 11: 16, 15: 14, 16: 19, 17: 14}}, 5: {
    'Smoothness': {1: 13, 4: {'AverageContrast': {8: 18, 6: 11, 7: 11}}, 5: 12,
                   6: {'AverageContrast': {9: {'Lobedness': {11: 12, 14: 20}}, 10: 16}}, 7: 17, 9: 16, 20: 14}}, 6: {
    'Smoothness': {2: 15, 3: 11, 4: 15, 5: 12, 6: {'AverageContrast': {9: 21, 10: 15}}, 17: 14}},
                                 7: {'Smoothness': {5: 19, 6: 14}},
                                 8: {'Smoothness': {2: {'Lobedness': {8: 10, 7: 9}}, 4: 20, 5: 18, 6: 14}},
                                 9: {'Smoothness': {3: 13, 5: {'MaximalIndentationDepth': {0: 15, 3: 16}}}},
                                 10: {'Smoothness': {8: 19, 2: 13, 4: 14, 6: {'AverageContrast': {10: 15, 11: 16}}}},
                                 12: {'Smoothness': {5: 17, 6: 20}}, 13: 18, 16: 21, 20: 20}}

header = 'Class, Specimen Number, Eccentricity, Aspect Ratio, Elongation, Solidity, Stochastic Convexity, Isoperimetric\
          Factor, Maximal Indentation Depth, Lobedness, Average Intensity, Average Contrast, Smoothness, Third moment,\
          Uniformity, Entropy'.replace(' ', '').split(',')
header_fast_list = {'Eccentricity': 0, 'AspectRatio': 1, 'Elongation': 2, 'Solidity': 3, 'StochasticConvexity': 4,
                    'IsoperimetricFactor': 5, 'MaximalIndentationDepth': 6, 'Lobedness': 7, 'AverageIntensity': 8,
                    'AverageContrast': 9, 'Smoothness': 10, 'Thirdmoment': 11, 'Uniformity': 12, 'Entropy': 13,
                    'Class': 14}

import dicttoxml
from xml.dom.minidom import parseString


def tree_to_xml(filename='tree', Tree=Tree):
    import json
    r = Tree
    r = json.dumps(r)
    Tree = json.loads(r)
    # 这么做是把int 的节点转成 str 的
    d = Tree
    xml = dicttoxml.dicttoxml(d, attr_type=False, root=False)
    # print(xml)
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
            return result  # if result == line[-1] else False
    return result  # if result == line[-1] else False


for_train_data = pd.read_csv('./for test.csv', names=header, sep=',')
# print(for_train_data.values)
for_train_data.drop('Specimen Number'.replace(' ', '').split(','), axis=1, inplace=True)
class_column = for_train_data.pop('Class')
for_train_data.insert(header.__len__() - 2, header[0], class_column)
step_array_define_by_train_array = np.array([
    0.0440665, 0.79127, 0.041884, 0.0242135,
    0.0301755, 0.0389892, 0.009807175, 0.360236785,
    0.007201405, 0.01156925, 0.003220385, 0.001477829,
    0.000146352, 0.1241515, 1])
step_array_define_by_test_array = np.array([
    0.036374, 0.90131, 0.040148, 0.025245,
    0.0301755, 0.0381162, 0.00924359, 0.32332322,
    0.00926314, 0.01221495, 0.003587885, 0.001249964,
    9.15038E-05, 0.12455, 1
])
step_array = step_array_define_by_test_array
lisanhua_data = for_train_data.values / step_array
lisanhua_data = np.array(lisanhua_data, dtype='int32')
print(lisanhua_data)

header.remove("SpecimenNumber")
header.append(header.pop(0))


def cal_accuracy(lisanhua_data, filename):
    cfrac_1_param = 0
    cfrac_2_param = lisanhua_data.__len__() + 0.0
    tree = ET.parse(filename)
    result_column = []

    for line in lisanhua_data:
        line = line.tolist()
        l_st_data = line[header_fast_list[tree.getroot().tag]]
        result = curretion(line, l_st_data, tree, line)
        try:

            if int(line[-1]) == int(result):
                # print(cfrac_1_param)
                cfrac_1_param += 1
                result_column.append(int(result))
            else:
                result_column.append(int(result))
        except:
            # pass
            # print(result)
            # print(line)
            result_column.append(int(result[-1]))
            # print(line[-1])
            if int(line[-1]) == int(result[-1]):
                cfrac_1_param += 1

    # print(cfrac_1_param)
    # print("[-] Header {}".format(header))
    print("[*] 准确率 {}".format(cfrac_1_param / cfrac_2_param))
    print("[-] Result_column {}".format(result_column))


import xml.etree.ElementTree as ET

if __name__ == '__main__':
    print('ID3 方法结果::')
    filename = 'leaf_tree_dict_by_huang.xml'
    tree_to_xml(filename=filename, Tree=Tree_by_huang2)
    cal_accuracy(lisanhua_data, filename)

    print('C4.5 方法结果::')
    filename = 'leaf_tree_dict.xml'
    tree_to_xml(filename=filename, Tree=Tree)
    cal_accuracy(lisanhua_data, filename)


    # tree_to_xml(filename=filename,Tree=Tree_by_huang)

    # cal_accuracy(lisanhua_data)
    # tree = ET.parse('leaf_tree_dict.xml')
    # cfrac_1_param = 0
    # line = [22, 2, 11, 38, 33, 20, 0, 0, 0, 3, 0, 0, 0, 2, 1]
    # line = line
    # l_st_data = line[header_fast_list[tree.getroot().tag]]
    # result = curretion(line, l_st_data, tree, line)
    # print(result)
# print(cfrac_1_param)
# tree_to_xml(Tree)

# i.tag n19 i.text None
# i.tag n18 i.text 36
