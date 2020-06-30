from Model import Model
import numpy as np

# 创建 Model， stride就是代表历史长度，通过更改model.stride就可以实现对模型的选择
# 例如: model.stride = n（n的取值范围5,10,15,20,25,30,35,40,45,50）
model = Model(stride=30)

#全连接神经网络 模型预测
# data_x是输入，输入的尺寸是（n, 6)，n是数据的个数
data_x = [[1,2,3,4,5,6]]

# data_y是输出的预测的结果，输出的矩阵的形状是(n, 4), n是数据的个数
data_y = model.predict_Net(data_x)
print(data_y)

# LSTM 模型预测
# data_x是输入，输入的矩阵形状是（n,stride*6)  stride是步长，n是数据的个数
data_x = [[1,2,3,4,5,6]]

# data_y是输出的预测的结果，输出的矩阵的形状是(4,n), n是数据的个数
data_y = model.predict_LSTM(data_x)
print(data_y)