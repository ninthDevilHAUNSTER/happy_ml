from shaobaobaoer_math_lab import mean, maximum_likelihood_estimation, stdev
#  main refer https://zhuanlan.zhihu.com/p/50432533
import numpy as np

w1 = np.array([[0.28, 1.31, -6.2], [0.07, 0.58, -0.78], [1.54, 2.01, -1.63],
               [-0.44, 1.18, -4.32], [-0.81, 0.21, 5.73], [1.52, 3.16, 2.77], [2.2, 2.42, -0.19],
               [0.91, 1.94, 6.21], [0.65, 1.93, 4.38], [-0.26, 0.82, -0.96]])
w2 = np.array([[0.011, 1.03, -0.21], [1.27, 1.28, 0.08], [0.13, 3.12, 0.16],
               [-0.21, 1.23, -0.11], [-2.18, 1.39, -0.19], [0.34, 1.96, -0.16], [-1.38, 0.94, 0.45],
               [-0.12, 0.82, 0.17], [-1.44, 2.31, 0.14], [0.26, 1.94, 0.08]])
w3 = np.array([[1.36, 2.17, 0.14], [1.41, 1.45, -0.38], [1.22, 0.99, 0.69],
               [2.46, 2.19, 1.31], [0.68, 0.79, 0.87], [2.51, 3.22, 1.35], [0.6, 2.44, 0.92],
               [0.64, 0.13, 0.97], [0.85, 0.58, 0.99], [0.66, 0.51, 0.88]])
ls1 = [w1, w2, w3]

from pprint import pprint
import math


class ParzenWinWithTwoClassification(object):
    def __init__(self):
        self.x1 = np.array([0.5, 1.0, 0.0])
        self.x2 = np.array([0.31, 1.51, -0.5])
        self.x3 = np.array([-0.3, 0.44, -0.1])
        self.ls2 = [self.x1, self.x2, self.x3]

    def pi(self, x, xi, hn):
        '''
        回算某一点的概率
        :param x:
        :param xi:
        :param hn:
        :return:
        '''
        print(x,xi,hn)
        row = x.shape[0]
        pi = 0
        for i in range(row):
            h = hn / math.sqrt(row)
            v = 4 * math.pi * math.pow(h, 3) / 3
            tp = x[i] - xi
            pi = pi + math.exp(-np.dot(tp.T, tp) / 2 * math.pow(h, 2) / v)
        print(pi/row)
        return pi / row

    def run_alg(self):
        ls1 = [w1, w2, w3]
        ls2 = self.ls2
        print('==========当h=1时============')
        for xi in ls2:
            classification = 0
            max_pi = 0
            for wi in range(len(ls1)):
                current_pi = self.pi(ls1[wi], xi, 1)
                if current_pi > max_pi:
                    classification = wi + 1
                    max_pi = current_pi
            print('样本点{0}属于{1}类'.format(xi, classification))


if __name__ == '__main__':
    ParzenWinWithTwoClassification().run_alg()
