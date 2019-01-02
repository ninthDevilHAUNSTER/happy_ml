# elipse_data_box
import numpy as np


class ElipseData(object):
    def __init__(self, u, v, a, b, curve_name, curve_color, class_label):
        # x, y = ellipse(4, 0, u * 0.1, 2.5)
        # long_list1.append([x, y, [1.0] * x.__len__(), 'b', 'ellipse x+ l'])
        self.x, self.y = self.ellipse(u, v, a, b)
        self.color = curve_color
        self.name = curve_name
        self.class_label = class_label

    def ellipse(self, u, v, a, b):
        # u = 1.  # x-position of the center
        # v = 0.5  # y-position of the center
        # a = 2.  # radius on the x-axis
        # b = 1.5  # radius on the y-axis

        t = np.linspace(0, 2 * np.pi, 100)
        return u + a * np.cos(t), v + b * np.sin(t)


# E_Data1 = ElipseData(u, v, a, b, curve_name, curve_color, class_label)
# E_Date2 = ElipseData(pass)