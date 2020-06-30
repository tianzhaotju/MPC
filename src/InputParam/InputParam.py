from .Blur import Blur
from .Cost import Cost
import math

'''
进料量：0~100 kg/h
进气量：9.5~16.5 m3/h
气化温度：360~850 ℃
H2：4~12 %
CH4：1~4 %
CO：10~28%
'''

class InputParam():
	def __init__(self):
		#用户输入
		self.table_burden = [
							[0, 50],
							[6, 70],
							[12, 90],
							[18, 130],
							[24, 170],
							[30, 190],
							[36, 230],
							[42, 210],
							[48, 180],
							[54, 150],
							[60, 170],
							[66, 190],
							[72, 180],
							[78, 160],
							[84, 150],
							]
		self.table_burden = [[0, 50], [10, 60], [20, 50]]

		self.table_burden = [
							[0, 20],
							[10, 35],
							[23, 55],
							[41, 45],
							[52, 60],
							[68, 40],
							[84, 65],
							[95, 50],
							[100, 55],
							]

		self.table_H2 = [
							[0, 6],
							[10, 7],
							[23, 8],
							[41, 8],
							[52, 6],
							[68, 7],
							[84, 8],
							[95, 8],
							[100, 7],
							]

		self.table_COCO2 = [
							[0, 2],
							[10, 2.2],
							[23, 1.8],
							[41, 1.8],
							[52, 2],
							[68, 2],
							[84, 2.2],
							[95, 2],
							[100, 2],
							]

		self.table_CQRZ = [
							[0, 4.2],
							[10, 4.3],
							[23, 4.5],
							[41, 4.2],
							[52, 4.3],
							[68, 4.3],
							[84, 4.5],
							[95, 4.2],
							[100, 4.3],
							]

		self.table_CH4 = [
							[0, 2],
							[10, 2.2],
							[23, 2.5],
							[41, 2],
							[52, 1.8],
							[68, 2],
							[84, 2.3],
							[95, 2.4],
							[100, 2.5],
							]

		self.range_intake_air = [9.5, 16.5]
		self.range_intake_material = [0, 100]
		self.range_derta_intake_air = [0, 5]
		self.range_derta_intake_material = [0, 20]
		self.weights_curve = [1, 1, 1, 1, 1]
		self.weights_perform = [0.33, 0.33, 0.33]
		self.weights_air_material = [0.7, 0.3]
		#用户输入的成本函数间的权重，默认为空（不输入）
		self.weights_cost = [1, 1, 1]
		#self.weights_cost = ''
		self.CHSON = [46.2, 5.1, 0.06, 35.4, 1.5]

		#内部变量
		#模糊控制的开始时间
		self.start_time = 0
		self.absolute_time = 0
		#模糊控制的时间段，单位min
		self.time_length_blur = 20
		self.blur = Blur()
		self.cost = Cost()
		#标准化区间
		self.standard_burden = [0, 75]
		self.standard_intake_air = [9.5, 16.5]
		self.standard_intake_material = [0, 100]
		self.standard_Tg = [360, 850]
		self.standard_H2 = [4, 12]
		self.standard_CH4 = [1, 4]
		self.standard_CO = [10, 28]
		self.standard_CO2 = [5, 25]
		self.standard_COCO2 = [10/25.0, 28/5.0]
		self.standard_CQL = [0.5, 3]
		self.standard_QHXL = [0.1, 0.9]
		self.standard_CQRZ = [3.5, 7]



	#获得进气量范围[9.5, 16.5]
	def get_range_intake_air(self):
		return self.range_intake_air

	#获得进料量范围[0, 100]
	def get_range_intake_material(self):
		return self.range_intake_material

	#获得进气量变化范围[0, 5]
	def get_range_derta_intake_air(self):
		return self.range_derta_intake_air

	#获得进料量变化范围[0, 20]
	def get_range_derta_intake_material(self):
		return self.range_derta_intake_material

	#设置开始时间
	def set_start_time(self, settime):
		self.start_time = settime + self.absolute_time
		#print(self.start_time)
		return True

	def add_absolute_time(self, settime):
		self.absolute_time = self.absolute_time + settime
		return True

	#获得用于模糊控制的Yref和P,
	def __get_dertaY_P(self):
		present_burdens = []
		lborder_time = self.start_time
		rborder_time = self.start_time + self.time_length_blur
		for burden in self.table_burden:
			if lborder_time <= burden[0] and burden[0] <= rborder_time:
				present_burdens.append(burden)

		maxY = 0
		minY = 1000
		P = 0
		for i in range(len(present_burdens)):
			if present_burdens[i][1] < minY:
				minY = present_burdens[i][1]
			if present_burdens[i][1] > maxY:
				maxY = present_burdens[i][1]
			if i != 0 and i != len(present_burdens) - 1:
				lburden = present_burdens[i-1][1]
				mburden = present_burdens[i][1]
				rburden = present_burdens[i+1][1]
				#极大
				if mburden > lburden and mburden > rburden:
					P = P+1
				#极小
				if mburden < lburden and mburden < rburden:
					P = P+1

		Y = maxY - minY
		#Y = 180
		#P = 2
		#print("△Y: ", Y)
		#print("P: ", P)
		return Y, P

	#预测时域Hp、跟踪误差成本权重Q
	def __get_Hp_Q(self):
		Y, P = self.__get_dertaY_P()
		Hp, Q = self.blur.get_Hp_Q(Y, P)
		#print(Hp)
		return Hp, Q

	#得到控制时域
	def get_control_time(self):
		Hp, Q = self.__get_Hp_Q()
		control_time = Hp
		return control_time

	def __get_demand_burden(self, _time):
		for i in range(len(self.table_burden)):
			if self.table_burden[i][0] == _time:
				return self.table_burden[i][1]
			elif i == len(self.table_burden) - 1:
				return self.table_burden[i][1]
			elif self.table_burden[i][0] < _time and _time < self.table_burden[i+1][0]:
				time1 = self.table_burden[i][0]
				time2 = self.table_burden[i+1][0]
				value1 = self.table_burden[i][1]
				value2 = self.table_burden[i+1][1]
				return value1 + (value2 - value1) * (_time - time1) / (time2 - time1)

	def __get_demand_H2(self, _time):
		for i in range(len(self.table_H2)):
			if self.table_H2[i][0] == _time:
				return self.table_H2[i][1]
			elif i == len(self.table_H2) - 1:
				return self.table_H2[i][1]
			elif self.table_H2[i][0] < _time and _time < self.table_H2[i+1][0]:
				time1 = self.table_H2[i][0]
				time2 = self.table_H2[i+1][0]
				value1 = self.table_H2[i][1]
				value2 = self.table_H2[i+1][1]
				return value1 + (value2 - value1) * (_time - time1) / (time2 - time1)

	def __get_demand_COCO2(self, _time):
		for i in range(len(self.table_COCO2)):
			if self.table_COCO2[i][0] == _time:
				return self.table_COCO2[i][1]
			elif i == len(self.table_COCO2) - 1:
				return self.table_COCO2[i][1]
			elif self.table_COCO2[i][0] < _time and _time < self.table_COCO2[i+1][0]:
				time1 = self.table_COCO2[i][0]
				time2 = self.table_COCO2[i+1][0]
				value1 = self.table_COCO2[i][1]
				value2 = self.table_COCO2[i+1][1]
				return value1 + (value2 - value1) * (_time - time1) / (time2 - time1)

	def __get_demand_CQRZ(self, _time):
		for i in range(len(self.table_CQRZ)):
			if self.table_CQRZ[i][0] == _time:
				return self.table_CQRZ[i][1]
			elif i == len(self.table_CQRZ) - 1:
				return self.table_CQRZ[i][1]
			elif self.table_CQRZ[i][0] < _time and _time < self.table_CQRZ[i+1][0]:
				time1 = self.table_CQRZ[i][0]
				time2 = self.table_CQRZ[i+1][0]
				value1 = self.table_CQRZ[i][1]
				value2 = self.table_CQRZ[i+1][1]
				return value1 + (value2 - value1) * (_time - time1) / (time2 - time1)

	def __get_demand_CH4(self, _time):
		for i in range(len(self.table_CH4)):
			if self.table_CH4[i][0] == _time:
				return self.table_CH4[i][1]
			elif i == len(self.table_CH4) - 1:
				return self.table_CH4[i][1]
			elif self.table_CH4[i][0] < _time and _time < self.table_CH4[i+1][0]:
				time1 = self.table_CH4[i][0]
				time2 = self.table_CH4[i+1][0]
				value1 = self.table_CH4[i][1]
				value2 = self.table_CH4[i+1][1]
				return value1 + (value2 - value1) * (_time - time1) / (time2 - time1)

	def __get_weights(self):
		Hp, Q = self.__get_Hp_Q()
		T = (1-Q) / 2
		R = T
		if not self.weights_cost == '':
			Q = self.weights_cost[0]
			R = self.weights_cost[1]
			T = self.weights_cost[2]
		weights_Q = [[Q]]
		weights_R = [[self.weights_air_material[0], 0], [0, self.weights_air_material[1]]]
		weights_T = [self.weights_perform]
		return weights_Q, weights_R, weights_T, R, T

	#用于绘图控制时域和权重
	def plot_Hc_weights(self):
		Hp, Q = self.__get_Hp_Q()
		T = (1-Q) / 2
		R = T
		Hc = Hp
		if not self.weights_cost == '':
			Q = self.weights_cost[0]
			R = self.weights_cost[1]
			T = self.weights_cost[2]
		return Hc, Q, R, T

	def __normalized_value(self, value, range):
		return (value - range[0]) / (range[1] - range[0])

	def __normalized_derta(self, derta, range):
		return derta / (range[1] - range[0])

	def get_product_burden(self, intake_material, intake_air, Tg, H2, CH4, CO):
		H2 = H2/100.0
		CH4 = CH4/100.0
		CO = CO/100.0
		#CO2=CO/2.18/e^(-450.893/(Tg+273))
		CO2 = CO / 2.18 / math.exp(-450.893/(Tg+273))
		N2 = 1 - CO2 - H2 - CH4 - CO
		CQRZ = 12.6*CO + 10.8*H2 + 35.9*CH4
		CQL = (intake_material + 1.29*intake_air) / (44.6*
		(H2*2 + CH4*16 + CO*28 + CO2*44 + N2*28)/1000) / intake_material
		HHV = 0.3491*self.CHSON[0] + 1.1783*self.CHSON[1] + 0.1005*self.CHSON[2] - 0.1034*self.CHSON[3] - 0.0151*self.CHSON[4]
		YLRZ = HHV - 2.256*9*self.CHSON[1]/100
		QHXL = CQRZ * CQL / YLRZ

		product_burden = CQRZ * CQL * intake_material / 3.6

		return product_burden, CQRZ, QHXL

	def get_product_COCO2(self, T):
		return 2.18 * math.exp(-450.893/(T + 273))

	def get_demand_burden(self, _time):
		return self.__get_demand_burden(_time)

	def get_demand_H2(self, _time):
		return self.__get_demand_H2(_time)

	def get_demand_COCO2(self, _time):
		return self.__get_demand_COCO2(_time)

	def get_demand_CQRZ(self, _time):
		return self.__get_demand_CQRZ(_time)

	def get_demand_CH4(self, _time):
		return self.__get_demand_CH4(_time)

	def __get_C1(self, product_burden, product_H2, product_COCO2, product_CQRZ, product_CH4):
		demand_burden = self.__get_demand_burden(self.start_time)
		demand_H2 = self.__get_demand_H2(self.start_time)
		demand_COCO2 = self.__get_demand_COCO2(self.start_time)
		demand_CQRZ = self.__get_demand_CQRZ(self.start_time)
		demand_CH4 = self.__get_demand_CH4(self.start_time)

		derta_burden = product_burden - demand_burden
		derta_H2 = product_H2 - demand_H2
		derta_COCO2 = product_COCO2 - demand_COCO2
		derta_CQRZ = product_CQRZ - demand_CQRZ
		derta_CH4 = product_CH4 - demand_CH4
		_derta_burden = self.__normalized_derta(derta_burden, self.standard_burden)
		_derta_H2 = self.__normalized_derta(derta_H2, self.standard_H2)
		_derta_COCO2 = self.__normalized_derta(derta_COCO2, self.standard_COCO2)
		_derta_CQRZ = self.__normalized_derta(derta_CQRZ, self.standard_CQRZ)
		_derta_CH4 = self.__normalized_derta(derta_CH4, self.standard_CH4)

		a1 = self.weights_curve[0] * _derta_burden * _derta_burden
		a2 = self.weights_curve[1] * _derta_H2 * _derta_H2
		a3 = self.weights_curve[2] * _derta_COCO2 * _derta_COCO2
		a4 = self.weights_curve[3] * _derta_CQRZ * _derta_CQRZ
		a5 = self.weights_curve[4] * _derta_CH4 * _derta_CH4

		return a1 + a2 + a3 + a4 + a5

	def get_cost(self, intake_air, intake_material, H2, CH4, CO, Tg, derta_intake_air, derta_intake_material):

		product_burden, CQRZ, QHXL = self.get_product_burden(intake_material, intake_air, Tg, H2, CH4, CO)
		product_H2 = H2
		product_COCO2 = self.get_product_COCO2(Tg)
		product_CQRZ = CQRZ
		product_CH4 = CH4

		weights_Q, weights_R, weights_T, R, T = self.__get_weights()
		C1 = weights_Q[0][0] * self.__get_C1(product_burden, product_H2, product_COCO2, product_CQRZ, product_CH4)*10

		derta_intake_air = self.__normalized_derta(derta_intake_air, self.standard_intake_air)
		derta_intake_material = self.__normalized_derta(derta_intake_material, self.standard_intake_material)
		C2 = self.cost.get_control_cost([derta_intake_air, derta_intake_material], weights_R, R)*10

		LHV = CQRZ
		Tg = Tg
		effect = QHXL
		LHV = self.__normalized_value(LHV, self.standard_CQRZ)
		Tg = self.__normalized_value(Tg, self.standard_Tg)
		effect = self.__normalized_value(effect, self.standard_QHXL)
		C3 = self.cost.get_perform_cost(LHV, Tg, effect, weights_T, T)
		#print("H2: ", H2)
		#print("CH4: ", CH4)
		#print("CO: ", CO)
		#print("CO2: ", CO2)
		#print("N2: ", N2)
		#print("原料热值 : ", YLRZ)
		#print("产气热值: ", CQRZ)
		#print("产气率: ", CQL)
		#print("气化效率: ", QHXL)
		#print("跟踪误差成本: ", C1)
		#print("控制量变化幅值成本: ", C2)
		#print("工艺性能成本: ", C3)
		#return C1
		return C1 + C2 + C3























