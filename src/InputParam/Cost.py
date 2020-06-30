import numpy as np

class Cost():
	def __init__(self):
		self.weights_Q = np.mat([[0.33]])

		self.weights_R = np.mat([[0.5, 0], [0, 0.5]])

		self.T = 0.33
		self.weights_T = np.mat([[0.33, 0.33, 0.33]])

	#跟踪误差成本（负荷成本）,dertaY = [20]
	def get_burden_cost(self, dertaY, weights_Q):
		dertaY = np.array(dertaY)
		dertaY = np.mat(dertaY)
		weights_Q = np.mat(weights_Q)
		result = self.get_norm_2(dertaY, weights_Q)
		result = float(result[0][0])
		return result

	#控制量变化幅值成本, dertaU = [2, 1]
	def get_control_cost(self, dertaU, weights_R, T):
		dertaU = np.array(dertaU)
		dertaU = np.mat(dertaU)
		weights_R = np.mat(weights_R)
		weights_R = weights_R * T
		result = self.get_norm_2(dertaU, weights_R)
		result = float(result[0][0])
		return result

	#工艺性能成本
	def get_perform_cost(self, LHV, Tg, effect, weights_T, T):
		weights_T = np.mat(weights_T)
		temp = np.mat([[LHV, Tg, effect]])
		result = T / temp * self.weights_T.T
		result = float(result[0][0])
		return result

	#参数都为np.matrix
	def get_norm_2(self, main_matrix, param_matrix):
		#print(np.shape(main_matrix), np.shape(param_matrix))
		result = main_matrix * param_matrix * main_matrix.T
		return result

















