from Network import Model
from PSO import optimize
from InputParam import InputParam
from GUI.DealResult import show_control,show_HcQ, show_burden, show_H2, show_COCO2, show_CQRZ, show_CH4, show_perform
import numpy as np
import time


def init_pso_result(best_x):
	_best_x = np.array([best_x[0]])
	_best_x = np.append(_best_x, _best_x, axis=0)
	#print(type(best_x))
	for item in best_x[1:]:
		item = np.array([item])
		_best_x = np.append(_best_x, item, axis=0)
		_best_x = np.append(_best_x, item, axis=0)
	return _best_x
#30 5 40 25
def predict_LSTM(model, pre_data, intake_material, intake_air):
		train_T = pre_data[-30:, :]
		train_H2 = pre_data[-5:, :]
		train_CH4 = pre_data[-40:, :]
		train_CO = pre_data[-25:, :]
		model.stride = 30
		T = model.predict_LSTM(train_T)[0][0]
		model.stride = 5
		H2 = model.predict_LSTM(train_H2)[0][1]
		model.stride = 40
		CH4 = model.predict_LSTM(train_CH4)[0][2]
		model.stride = 25
		CO = model.predict_LSTM(train_CO)[0][3]

		new_data = np.array([[intake_material, intake_air, T, H2, CH4, CO]])
		return np.append(pre_data, new_data, axis=0)

def get_result_LSTM(model, pre_data, best_x):
	_best_x = init_pso_result(best_x)
	for i in range(0, len(_best_x)):
		pre_data = predict_LSTM(model, pre_data, _best_x[i][0], _best_x[i][1])
	return pre_data
#30 15 40 10
def predict_Net(model, pre_data, intake_material, intake_air):
		train_T = pre_data[-30:, :]
		train_H2 = pre_data[-15:, :]
		train_CH4 = pre_data[-40:, :]
		train_CO = pre_data[-10:, :]
		model.stride = 30
		T = model.predict_Net(train_T)[0][0]
		model.stride = 15
		H2 = model.predict_Net(train_H2)[0][1]
		model.stride = 40
		CH4 = model.predict_Net(train_CH4)[0][2]
		model.stride = 10
		CO = model.predict_Net(train_CO)[0][3]

		new_data = np.array([[intake_material, intake_air, T, H2, CH4, CO]])
		return np.append(pre_data, new_data, axis=0)

def get_result_Net(model, pre_data, best_x):
	_best_x = init_pso_result(best_x)
	for i in range(0, len(_best_x)):
		pre_data = predict_Net(model, pre_data, _best_x[i][0], _best_x[i][1])
	length = len(_best_x)
	return pre_data[-length:]


data = np.loadtxt('./data/data.csv', delimiter=',', skiprows=2)[:, 1: 7]
model = Model(stride=30)
param = InputParam()
'''
best_x = [[28.12421607,14.41792793],
[74.20071455, 9.92461491],
[58.43944117, 11.0825983 ],
[69.17058085, 10.75863371],
[36.35079316, 11.28547075],
[57.59862793, 10.92375781],
[27.67301199, 14.30238826],
[75.87876121, 13.22701089],
[100.0, 16.10638031],
[100.0, 13.78599625]]

best_y = 26.45409999049893
'''

start = time.clock()
net_data = data
for i in range(0, 5):
	print(i)
	param.set_start_time(0)
	Hc = param.get_control_time()
	Hc = int(Hc)
	best_x, best_y = optimize(Hc, data, 2, model, param, 15)
	best_x = np.array(best_x)

	#print('best_x = ', best_x)
	print('best_y = ', best_y)

	new_net_data = get_result_Net(model, data, best_x)
	net_data = np.append(net_data, new_net_data, axis=0)
	data = get_result_LSTM(model, data, best_x)
	#result = abs((data - net_data) / data)
	#print(np.mean(result, axis=0))
	#设置绝对时长
	time_length = len(new_net_data)
	param.add_absolute_time(time_length)
	i = i + 1
end = time.clock()
print('time=', end - start)
show_control(data[60:])
show_HcQ(param, data[60:])
show_burden(param, net_data[60:], data[60:])
show_H2(param, net_data[60:], data[60:])
show_COCO2(param, net_data[60:], data[60:])
show_CQRZ(param, net_data[60:], data[60:])
show_CH4(param, net_data[60:], data[60:])
show_perform(param, data[60:])










