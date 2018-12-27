# 数据集合说明

## OVR SVMs

这里帮你搞了搞非常好用的数据集！

- data_loader/gener/ellipse_data_easy_to_splict.txt
- data_loader/gener/ellipse_data_easy_to_splict2.txt

这个数据集的椭圆分得非常开，做分类当然可以达到100%的准确率！

相关参数
```python
        x, y = ellipse(8, 0, u * 0.1, 4)
        long_list.append([x, y, [1.0] * x.__len__(), 'b', 'ellipse x+ l'])
        x, y = ellipse(0, 8, 4, u * 0.1)
        long_list.append([x, y, [2.0] * x.__len__(), 'r', 'ellipse y+ -'])
        x, y = ellipse(-8, 0, u * 0.1, 4)
        long_list.append([x, y, [3.0] * x.__len__(), 'g', 'ellipse x- l'])
        x, y = ellipse(0, -8, 4, u * 0.1)
        long_list.append([x, y, [4.0] * x.__len__(), 'y', 'ellipse y- -'])
```