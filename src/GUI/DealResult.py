# coding: utf-8
import os, sys
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
plt.rcParams['font.sans-serif']=['SimHei']	#画图显示中文
import pandas as pd
import tushare as ts
import numpy as np

#data = np.loadtxt('../data/data.csv', delimiter=',', skiprows=2)[:, 1: 7]

def show_control(data):
	#处理数据
	data = pd.DataFrame(data)
	data.columns = ['进料量', '进气量', '气化温度', 'H2%', 'CH4%', 'CO%']

	#画图
	fig,axes = plt.subplots(2,1)
	data[['进气量']].plot(ax=axes[0], grid=True, use_index=True)
	data[['进料量']].plot(ax=axes[1], grid=True, use_index=True)
	axes[0].yaxis.set_major_formatter(FormatStrFormatter('%d m3/h'))
	axes[1].yaxis.set_major_formatter(FormatStrFormatter('%d kg/h'))
	plt.suptitle('控制量随时间的变化', fontsize=15)
	plt.xlabel('min', fontsize=15)
	plt.show()

def show_HcQ(param, data):
	Hc_data = np.array([[0]])
	weights_data = np.array([[0, 0, 0]])
	for i in range(0, len(data)):
		param.set_start_time(i)
		Hc, Q, R, T = param.plot_Hc_weights()
		Hc_data = np.append(Hc_data, np.array([[Hc]]), axis=0)
		weights_data = np.append(weights_data, np.array([[Q, R, T]]), axis=0)
	Hc_data = Hc_data[1:]
	Hc_data = pd.DataFrame(Hc_data)
	Hc_data.columns = ['控制时域']

	weights_data = weights_data[1:]
	weights_data = pd.DataFrame(weights_data)
	weights_data.columns = ['成本函数1：控制参考曲线', '成本函数2：控制量变化幅值', '成本函数3：工艺性能']

	#画图
	fig,axes = plt.subplots(1,1)
	Hc_data.plot(ax=axes, grid=True, use_index=True)
	plt.suptitle('控制时域变化情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	plt.gca().yaxis.set_major_formatter(FuncFormatter(Hc_unit))

	fig,axes = plt.subplots(1,1)
	weights_data.plot(ax=axes, grid=True, use_index=True)
	plt.suptitle('成本函数权重变化情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	plt.show()

def show_burden(param, net, lstm):
	#负荷数据
	burden_data = np.array([[0, 0, 0]])
	#误差数据
	derta_data = np.array([[0, 0, 0]])
	for i in range(0, len(net)):
		demand_burden = param.get_demand_burden(i)
		net_burden = param.get_product_burden(net[i][0], net[i][1], net[i][2], net[i][3], net[i][4], net[i][5])[0]
		lstm_burden = param.get_product_burden(lstm[i][0], lstm[i][1], lstm[i][2], lstm[i][3], lstm[i][4], lstm[i][5])[0]
		#net与demand
		derta1 = rela_error(net_burden, demand_burden)
		#lstm与net
		derta2 = rela_error(lstm_burden, net_burden)
		#demand与lstm
		derta3 = rela_error(demand_burden, lstm_burden)
		burden_data = np.append(burden_data, np.array([[demand_burden, net_burden, lstm_burden]]), axis=0)
		derta_data = np.append(derta_data, np.array([[derta1, derta2, derta3]]), axis=0)
	burden_data = burden_data[1:]
	burden_data = pd.DataFrame(burden_data)
	burden_data.columns = ['用户输入', 'Network', 'LSTM']

	derta_data = derta_data[1:]
	derta_data = pd.DataFrame(derta_data)
	derta_data.columns = ['Network与用户输入误差', 'LSTM与Network误差', '用户输入与LSTM误差']

	#画图
	fig,axes = plt.subplots(2,1)
	burden_data.plot(ax=axes[0], grid=True, use_index=True)
	plt.suptitle('负荷曲线跟踪情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	axes[0].yaxis.set_major_formatter(FuncFormatter(burden_unit))
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(burden_unit))
	derta_data.plot(ax=axes[1], grid=True, use_index=True)
	axes[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.show()

def rela_error(a, b):
	if a > b:
		return (a - b) / a
	return (b - a) / b

def show_H2(param, net, lstm):
	#H2数据
	value_data = np.array([[0, 0, 0]])
	#误差数据
	derta_data = np.array([[0, 0, 0]])
	for i in range(0, len(net)):
		demand_value = param.get_demand_H2(i)
		net_value = net[i][3]
		lstm_value = lstm[i][3]
		#net与demand
		derta1 = rela_error(net_value, demand_value)
		#lstm与net
		derta2 = rela_error(lstm_value, net_value)
		#demand与lstm
		derta3 = rela_error(demand_value, lstm_value)
		value_data = np.append(value_data, np.array([[demand_value, net_value, lstm_value]]), axis=0)
		derta_data = np.append(derta_data, np.array([[derta1, derta2, derta3]]), axis=0)
	value_data = value_data[1:]
	value_data = pd.DataFrame(value_data)
	value_data.columns = ['用户输入', 'Network', 'LSTM']

	derta_data = derta_data[1:]
	derta_data = pd.DataFrame(derta_data)
	derta_data.columns = ['Network与用户输入误差', 'LSTM与Network误差', '用户输入与LSTM误差']

	#画图
	fig,axes = plt.subplots(2,1)
	value_data.plot(ax=axes[0], grid=True, use_index=True)
	plt.suptitle('H2%曲线跟踪情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	axes[0].yaxis.set_major_formatter(FuncFormatter(add_percent))
	derta_data.plot(ax=axes[1], grid=True, use_index=True)
	axes[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.show()

def show_COCO2(param, net, lstm):
	#CO/CO2数据
	value_data = np.array([[0, 0, 0]])
	#误差数据
	derta_data = np.array([[0, 0, 0]])
	for i in range(0, len(net)):
		demand_value = param.get_demand_COCO2(i)
		net_value = param.get_product_COCO2(net[i][2])
		lstm_value = param.get_product_COCO2(lstm[i][2])
		#net与demand
		derta1 = rela_error(net_value, demand_value)
		#lstm与net
		derta2 = rela_error(lstm_value, net_value)
		#demand与lstm
		derta3 = rela_error(demand_value, lstm_value)
		value_data = np.append(value_data, np.array([[demand_value, net_value, lstm_value]]), axis=0)
		derta_data = np.append(derta_data, np.array([[derta1, derta2, derta3]]), axis=0)
	value_data = value_data[1:]
	value_data = pd.DataFrame(value_data)
	value_data.columns = ['用户输入', 'Network', 'LSTM']

	derta_data = derta_data[1:]
	derta_data = pd.DataFrame(derta_data)
	derta_data.columns = ['Network与用户输入误差', 'LSTM与Network误差', '用户输入与LSTM误差']

	#画图
	fig,axes = plt.subplots(2,1)
	value_data.plot(ax=axes[0], grid=True, use_index=True)
	plt.suptitle('CO/CO2曲线跟踪情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	#axes[0].yaxis.set_major_formatter(FuncFormatter(to_percent))
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(burden_unit))
	derta_data.plot(ax=axes[1], grid=True, use_index=True)
	axes[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.show()

def show_CQRZ(param, net, lstm):
	#产气热值数据
	value_data = np.array([[0, 0, 0]])
	#误差数据
	derta_data = np.array([[0, 0, 0]])
	for i in range(0, len(net)):
		demand_value = param.get_demand_CQRZ(i)
		net_value = param.get_product_burden(net[i][0], net[i][1], net[i][2], net[i][3], net[i][4], net[i][5])[1]
		lstm_value = param.get_product_burden(lstm[i][0], lstm[i][1], lstm[i][2], lstm[i][3], lstm[i][4], lstm[i][5])[1]
		#net与demand
		derta1 = rela_error(net_value, demand_value)
		#lstm与net
		derta2 = rela_error(lstm_value, net_value)
		#demand与lstm
		derta3 = rela_error(demand_value, lstm_value)
		value_data = np.append(value_data, np.array([[demand_value, net_value, lstm_value]]), axis=0)
		derta_data = np.append(derta_data, np.array([[derta1, derta2, derta3]]), axis=0)
	value_data = value_data[1:]
	value_data = pd.DataFrame(value_data)
	value_data.columns = ['用户输入', 'Network', 'LSTM']

	derta_data = derta_data[1:]
	derta_data = pd.DataFrame(derta_data)
	derta_data.columns = ['Network与用户输入误差', 'LSTM与Network误差', '用户输入与LSTM误差']

	#画图
	fig,axes = plt.subplots(2,1)
	value_data.plot(ax=axes[0], grid=True, use_index=True)
	plt.suptitle('产气热值曲线跟踪情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	axes[0].yaxis.set_major_formatter(FuncFormatter(CQRZ_unit))
	derta_data.plot(ax=axes[1], grid=True, use_index=True)
	axes[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.show()

def show_CH4(param, net, lstm):
	#CH4%数据
	value_data = np.array([[0, 0, 0]])
	#误差数据
	derta_data = np.array([[0, 0, 0]])
	for i in range(0, len(net)):
		demand_value = param.get_demand_CQRZ(i)
		net_value = net[i][4]
		lstm_value = lstm[i][4]
		#net与demand
		derta1 = rela_error(net_value, demand_value)
		#lstm与net
		derta2 = rela_error(lstm_value, net_value)
		#demand与lstm
		derta3 = rela_error(demand_value, lstm_value)
		value_data = np.append(value_data, np.array([[demand_value, net_value, lstm_value]]), axis=0)
		derta_data = np.append(derta_data, np.array([[derta1, derta2, derta3]]), axis=0)
	value_data = value_data[1:]
	value_data = pd.DataFrame(value_data)
	value_data.columns = ['用户输入', 'Network', 'LSTM']

	derta_data = derta_data[1:]
	derta_data = pd.DataFrame(derta_data)
	derta_data.columns = ['Network与用户输入误差', 'LSTM与Network误差', '用户输入与LSTM误差']

	#画图
	fig,axes = plt.subplots(2,1)
	value_data.plot(ax=axes[0], grid=True, use_index=True)
	plt.suptitle('CH4%曲线跟踪情况', fontsize=15)
	plt.xlabel('min', fontsize=15)
	axes[0].yaxis.set_major_formatter(FuncFormatter(add_percent))
	derta_data.plot(ax=axes[1], grid=True, use_index=True)
	axes[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.show()

def show_perform(param, data):
	draw_data = np.array([[0, 0, 0]])
	for i in range(0, len(data)):
		burden, CQRZ, QHXL = param.get_product_burden(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5])
		draw_data = np.append(draw_data, np.array([[CQRZ, data[i][2], QHXL]]), axis=0)
	draw_data = draw_data[1:]
	draw_data = pd.DataFrame(draw_data)
	draw_data.columns = ['产气热值', '气化温度', '气化效率']
	#print(draw_data)
	#画图
	
	fig,axes = plt.subplots(3,1)
	plt.suptitle('工艺性能变化情况', fontsize=15)
	draw_data[['产气热值']].plot(ax=axes[0], grid=True, use_index=True)
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(CQRZ_unit))
	axes[0].yaxis.set_major_formatter(FuncFormatter(CQRZ_unit))
	draw_data[['气化温度']].plot(ax=axes[1], grid=True, use_index=True)
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(T_unit))
	axes[1].yaxis.set_major_formatter(FuncFormatter(T_unit))
	draw_data[['气化效率']].plot(ax=axes[2], grid=True, use_index=True)
	#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	axes[2].yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.xlabel('min', fontsize=15)
	plt.show()


def to_percent(temp, position):
		return '%.2f'%(100.0*temp) + '%'

def add_percent(temp, position):
		return '%.2f'%(temp) + '%'

def burden_unit(temp, position):
		return '%.1f'%(temp) + 'kW'

def CQRZ_unit(temp, position):
		return '%.2f'%(temp) + 'kJ/m3'

def T_unit(temp, position):
		return '%.1f'%(temp) + '℃'

def Hc_unit(temp, position):
		return '%d'%(temp) + 'min'













