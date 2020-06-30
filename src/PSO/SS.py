from .pso import PSO
import numpy as np
import random
import time
import matplotlib.pyplot as plt

#model_H2 = Model(stride=15)
#model_CH4 = Model(stride=40)
#model_CO = Model(stride=10)

# 返回矩阵为（n*3，8），最后两列是Δair和Δmaterial
def init_data(n, data, interval, model):
    cost_data = np.zeros((n, 2)) #n*3
    for i in range(n): #n*3
        if (i + 1) % interval: # 需要计算的时间点
            intake_material = random.uniform(0, 100)
            intake_air = random.uniform(9.5, 16.5)
            while not (-5 < data[-1, 1] - intake_air < 5 and -20 < data[-1, 0] - intake_material < 20):
                intake_material = random.uniform(0, 100)
                intake_air = random.uniform(9.5, 16.5)
            # 求间隔的变化量
            derta_intake_air = intake_air - data[-1, 1]
            derta_intake_material = intake_material - data[-1, 0]
            derta = np.array([derta_intake_air, derta_intake_material])
            cost_data[i] = derta
        else:
            intake_material = data[-1, 0]
            intake_air = data[-1, 1]


        train_T = data[-30:, :]
        train_H2 = data[-15:, :]
        train_CH4 = data[-40:, :]
        train_CO = data[-10:, :]
        model.stride = 30
        T = model.predict_Net(train_T)[0][0]
        model.stride = 15
        H2 = model.predict_Net(train_H2)[0][1]
        model.stride = 40
        CH4 = model.predict_Net(train_CH4)[0][2]
        model.stride = 10
        CO = model.predict_Net(train_CO)[0][3]

        new_data = np.array([intake_material, intake_air, T, H2, CH4, CO])
        data = np.vstack((data, new_data))

    opt_data = np.hstack((data[-n:, ], cost_data)) #n*3
    # print(opt_data.shape)
    return opt_data


#init_data(n, data)
# n是控制时域，interval是时间间隔
def optimize(n, data, interval, model, param, max_iter):
    pop = 20
    opt_data = init_data(n, data, interval, model)[np.newaxis, :, :]
    for i in range(pop - 1):
        opt_data = np.vstack((opt_data, init_data(n, data, interval, model)[np.newaxis, :, :]))

    data_x = opt_data[:, 0: 1, 0: 2]
    for i in range(1, int(opt_data.shape[1])): #opt_data.shape[1] / 3
        if i % interval == 0:
            data_x = np.hstack((data_x, opt_data[:, i: i+1, 0: 2]))

    pso = PSO(param=param,model=model, h_data=data, x=data_x, cost_data=opt_data, max_iter=max_iter,
              lb=np.array([1, 9.5]), ub=np.array([100, 16.5]), w=0.7, c1=0.4, c2=0.6)
    pso.run()
    #print('best_x is ', pso.gbest_x, '\nbest_y is', pso.gbest_y)
    #plt.plot(pso.gbest_y_hist)
    #plt.show()
    #print(pso.gbest_y_hist)
    return pso.gbest_x, pso.gbest_y
#start = time.clock()
# 调用optimize
#optimize(20, data, 2, model)
#end = time.clock()
#print('time=', end - start)
