import numpy as np
from .base import SkoBase

class PSO(SkoBase):

    def __init__(self, param, model, h_data, x, cost_data, max_iter=150, lb=None, ub=None, w=0.6, c1=0.5, c2=0.5):
        #self.func = func_transformer(func)
        self.param = param
        self.model = model
        self.h_data = h_data.copy()
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.X = x
        self.pop = self.X.shape[0]  # number of particles
        self.dim = self.X.shape[1:3]  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.cost_data = cost_data  # 用于计算成本函数

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        #self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))

        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim[0], self.dim[1]))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim[0], self.dim[1]))  # global best location for all particles
        #self.gbest_y = np.zeros((self.dim[0], 1))  # global best y for all particles
        #self.gbest_y[:, 0] = np.inf
        self.gbest_y = np.inf
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}


    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim[0], self.dim[1])
        r2 = np.random.rand(self.pop, self.dim[0], self.dim[1])
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)


    def update_X(self):
        h_data = self.h_data.copy()
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

        for i in range(self.pop):
            for j in range(self.X.shape[1]):
                self.cost_data[i, 2 * j, 0: 2] = self.X[i, j, 0: 2]
                self.cost_data[i, 2 * j + 1, 0: 2] = self.X[i, j, 0: 2]
                self.cost_data[i, j, 6] = self.cost_data[i, j + 1, 1] - self.cost_data[i, j, 1]
                self.cost_data[i, j, 7] = self.cost_data[i, j + 1, 0] - self.cost_data[i, j, 0]

        for i in range(self.pop):
            for j in range(self.X.shape[1] * 2): #*6
                train_T = h_data[-30:, :]
                train_H2 = h_data[-15:, :]
                train_CH4 = h_data[-40:, :]
                train_CO = h_data[-10:, :]
                self.model.stride = 30
                T = self.model.predict_Net(train_T)[0][0]
                self.model.stride = 15
                H2 = self.model.predict_Net(train_H2)[0][1]
                self.model.stride = 40
                CH4 = self.model.predict_Net(train_CH4)[0][2]
                self.model.stride = 10
                CO = self.model.predict_Net(train_CO)[0][3]
                self.cost_data[i, j, :] = np.array([self.cost_data[i, j, 0], self.cost_data[i, j, 1],
                                                    T, H2, CH4, CO, self.cost_data[i, j, 6], self.cost_data[i, j, 7]])
                new_data = np.array([self.cost_data[i, j, 0], self.cost_data[i, j, 1], T, H2, CH4, CO])
                h_data = np.vstack((h_data, new_data))

        #print(self.cost_data)
    """
    计算目标函数
    """
    def cal_y(self):
        # calculate y for every x in X
        # 传入L, G, C
        temp = np.zeros((self.pop, self.cost_data.shape[1], 1))
        self.Y = np.zeros((self.pop, 1))
        for i in range(self.pop):
            for j in range(self.cost_data.shape[1]): ##############
                intake_material = self.cost_data[i][j][0]
                intake_air = self.cost_data[i][j][1]
                T = self.cost_data[i][j][2]
                H2 = self.cost_data[i][j][3]
                CH4 = self.cost_data[i][j][4]
                CO = self.cost_data[i][j][5]
                derta_intake_air = self.cost_data[i][j][6]
                derta_intake_material = self.cost_data[i][j][7]
                self.param.set_start_time(j)
                cost = self.param.get_cost(intake_air, intake_material, H2, CH4, CO, T, derta_intake_air, derta_intake_material)
                temp[i][j][0] = cost
            self.Y[i][0] = np.sum(temp[i])
        #self.Y = self.func(self.X).reshape(-1, 1)
        #print(self.Y)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        #self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        for i in range(self.pbest_y.shape[0]):
            if self.pbest_y[i][0] > self.Y[i][0]:
                self.pbest_x[i] = self.X[i]
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)
        #print(self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        argmin = 0
        min_x = self.Y[0][0]
        for i in range(self.Y.shape[0]):
            if self.Y[i][0] < min_x:
                argmin = i
                min_x = self.Y[i][0]
        if self.gbest_y > min_x:
            self.gbest_x = self.X[argmin, :, :].copy()
            self.gbest_y = min_x
        #print(self.Y)

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    fit = run