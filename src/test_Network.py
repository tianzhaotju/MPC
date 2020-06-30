from Network import Model
import numpy as np
'''原Interface'''

data = np.loadtxt('./data/data.csv', delimiter=',', skiprows=2)[:, 1: 7]
# 创建 Model， stride就是代表历史长度，通过更改model.stride就可以实现对模型的选择
# 例如: model.stride = n（n的取值范围5,10,15,20,25,30,35,40,45,50）
model = Model(stride=30)

#全连接神经网络 模型预测
# data_x是输入，输入的尺寸是（n, 6)，n是数据的个数
model.stride = 25
data_x = data[-25:, :]

# data_y是输出的预测的结果，输出的矩阵的形状是(n, 4), n是数据的个数
data_y = model.predict_LSTM(data_x)
print(data_y)

# LSTM 模型预测
# data_x是输入，输入的矩阵形状是（n,stride*6)  stride是步长，n是数据的个数
#data_x = [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]

# data_y是输出的预测的结果，输出的矩阵的形状是(4,n), n是数据的个数
#model.stride = 5
data_y = model.predict_Net(data_x)
print(type(data_y[0]))


############################################
T_len, H2_len, CH4_len, CO_len = 5,10,5,20

# 训练LSTM & Net
for stride in [T_len, H2_len, CH4_len, CO_len]:
    model.online_train_Net(train_path='./data/data_1.csv',model_path='./newmodels',stride=stride, EPOCH=1000)
    model.online_train_LSTM(train_path='./data/data_1.csv',model_path='./newmodels',stride=stride, EPOCH=1000)

# 测试LSTM & Net，返回的res_Net和res_LSTM的格式一样：[气化温度误差率，H2误差率，CH4误差率，CO误差率，平均耗时]
res_Net = model.online_test_Net(test_path='./data/data_1.csv',model_path='./newmodels',stride=5)
res_LSTM = model.online_test_LSTM(test_path='./data/data_1.csv',model_path='./newmodels',stride=5)
#############################################
