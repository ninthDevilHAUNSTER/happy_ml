from pprint import pprint

# 1. 定义数据集
x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]

# 2. 使用Matplotlib 绘制数据
from matplotlib import pyplot as plt

plt.scatter(x, y)
plt.show()


# 3. 实现 2 次多项式函数及误差函数
def func(p, x1):
    w0, w1, w2 = p
    f = w0 + w1 * x1 + w2 * x1 * x1
    return f


def err_func(p, x1, y1):
    ret = func(p, x1) - y1
    return ret


# 4. 生成随机数
import numpy as np

# 生成 3 个随机数
p_init = np.random.randn(3)

p_init

# 5. 使用 Scipy 提供的最小二乘法哈数得到最佳拟合参数
from scipy.optimize import leastsq

parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))

print('Fitting Parameters: ', parameters[0])

## 6. 绘制拟合后的图像
# 绘制拟合图像时需要的临时点
x_temp = np.linspace(0, 80, 10000)

# 绘制拟合函数曲线
plt.plot(x_temp, func(parameters[0], x_temp), 'r')

# 绘制原数据点
plt.scatter(x, y)
