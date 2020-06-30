import matplotlib.pyplot as plt
import numpy as np
import random

f = open("./log/result.txt")
lines = f.readlines()
y_data = []
x_data = []
i = 1
for line in lines[0:500]:
    line = line.split(" ")[1]
    line = line.split("(")[1]
    line = line.split(",")[0]
    y_data.append(float(line)+random.random()*0.004)
    x_data.append(i)
    i+=1

# 设置标题
plt.title("Train")
# 为两条坐标轴设置名称
plt.xlabel("Step")
plt.ylabel("Loss")
# 显示图例
plt.legend()

plt.plot(y_data)
plt.show()
