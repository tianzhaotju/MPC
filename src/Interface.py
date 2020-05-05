from Model import Model
import numpy as np

# 创建 Model
model = Model(stride=5,EPOCH=1)


# Net模型预测
# data_x是输入，输入的尺寸是（stride*2,n)  stride是步长，n是数据的个数
data_x = [[1,1,2,2,3,3,4,4,5,5],[6,6,7,7,8,8,9,9,0,0]]
# data_y是输出的预测的结果，输出的矩阵的形状是(4,n), n是数据的个数
data_y = model.predict_Net(data_x)
print(data_y)


# LSTM模型预测
# data_x是输入，输入的矩阵形状是（stride*2,n)  stride是步长，n是数据的个数
data_x = [[1,1,2,2,3,3,4,4,5,5],[6,6,7,7,8,8,9,9,0,0]]
# data_y是输出的预测的结果，输出的矩阵的形状是(4,n), n是数据的个数
data_y = model.predict_LSTM(data_x)
print(data_y)
