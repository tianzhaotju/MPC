from Network import Model
from PSO import optimize
from InputParam import InputParam
import numpy as np
import time

data = np.loadtxt('./data/data.csv', delimiter=',', skiprows=2)[:, 1: 7]
model = Model(stride=30)
param = InputParam()

start = time.clock()
# 调用optimize
optimize(20, data, 2, model, param)
end = time.clock()