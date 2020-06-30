



class Blur():
	def __init__(self):
		self.range_blur_Y = [5, 10, 15, 20, 25, 30, 35]
		self.range_blur_P = [0, 0.5, 1, .15, 2, 2.5, 3]
		self.range_blur_sub = ['NB', 'NM', 'NS', 'ZO', 'PS', 'PM', 'PB']

		self.map_deblur_Hp = {
						'NB':8,
						'NM':10,
						'NS':12,
						'ZO':14,
						'PS':16,
						'PM':18,
						'PB':20}

		self.map_deblur_Q = {
						'NB':0.5,
						'NM':0.55,
						'NS':0.6,
						'ZO':0.65,
						'PS':0.7,
						'PM':0.75,
						'PB':0.8}

		self.map_sub_index = {
						'NB':0,
						'NM':1,
						'NS':2,
						'ZO':3,
						'PS':4,
						'PM':5,
						'PB':6}

		self.matrix_blur = [
							['NB','NB','NB','NB','NM','ZO','ZO'],
							['NB','NB','NB','NM','NM','ZO','ZO'],
							['NB','NM','NM','NS','ZO','PS','PM'],
							['NM','NM','NS','ZO','PS','PM','PM'],
							['NS','NS','ZO','PM','PM','PM','PB'],
							['ZO','ZO','ZO','PM','PB','PB','PB'],
							['ZO','PS','PB','PB','PB','PB','PB']]

		#[P][Y]
		self.matrix_blur = {
							'NB':{'NB':'NB', 'NM':'NB', 'NS':'NB', 'ZO':'NB', 'PS':'NM', 'PM':'ZO', 'PB':'ZO'},
							'NM':{'NB':'NB', 'NM':'NB', 'NS':'NB', 'ZO':'NM', 'PS':'NM', 'PM':'ZO', 'PB':'ZO'},
							'NS':{'NB':'NB', 'NM':'NM', 'NS':'NM', 'ZO':'NS', 'PS':'ZO', 'PM':'PS', 'PB':'PM'},
							'ZO':{'NB':'NM', 'NM':'NM', 'NS':'NS', 'ZO':'ZO', 'PS':'PS', 'PM':'PM', 'PB':'PM'},
							'PS':{'NB':'NS', 'NM':'NS', 'NS':'ZO', 'ZO':'PM', 'PS':'PM', 'PM':'PM', 'PB':'PB'},
							'PM':{'NB':'ZO', 'NM':'ZO', 'NS':'ZO', 'ZO':'PM', 'PS':'PB', 'PM':'PB', 'PB':'PB'},
							'PB':{'NB':'ZO', 'NM':'PS', 'NS':'PB', 'ZO':'PB', 'PS':'PB', 'PM':'PB', 'PB':'PB'}}


	def blur_Y(self, Y):
		l_sub = r_sub = ''
		a = b = 0 

		if Y <= self.range_blur_Y[0]:
			l_sub = r_sub = self.range_blur_sub[0]
			a = 0
			b = 1
		elif Y > self.range_blur_Y[6]:
			l_sub = r_sub = self.range_blur_sub[6]
			a = 1
			b = 0
		else:
			#1-6个区间判定
			index = 1
			#当不在此区间时
			while not(Y > self.range_blur_Y[index - 1] and Y <= self.range_blur_Y[index]):
				index = index + 1
			#print('Y', index)
			l_sub = self.range_blur_sub[index - 1]
			r_sub = self.range_blur_sub[index]
			temp_lY = self.range_blur_Y[index - 1]
			temp_rY = self.range_blur_Y[index]
			b = float(Y - temp_lY) / float(temp_rY - temp_lY)
			a = float(temp_rY - Y) / float(temp_rY - temp_lY)
		#a + b = 1, 表示左子集和右子集的隶属度
		return l_sub, r_sub, a, b

	def blur_P(self, P):
		l_sub = r_sub = ''
		a = b = 0 

		if P <= self.range_blur_P[0]:
			l_sub = r_sub = self.range_blur_sub[0]
			a = 0
			b = 1
		elif P > self.range_blur_P[6]:
			l_sub = r_sub = self.range_blur_sub[6]
			a = 1
			b = 0
		else:
			#1-6个区间判定
			index = 1
			#当不在此区间时
			while not(P > self.range_blur_P[index - 1] and P <= self.range_blur_P[index]):
				index = index + 1
			#print('P', index)
			l_sub = self.range_blur_sub[index - 1]
			r_sub = self.range_blur_sub[index]
			temp_lP = self.range_blur_P[index - 1]
			temp_rP = self.range_blur_P[index]
			b = float(P - temp_lP) / float(temp_rP - temp_lP)
			a = float(temp_rP - P) / float(temp_rP - temp_lP)
		return l_sub, r_sub, a, b

	#a是Y的左子集隶属度，b是P的左子集隶属度（ppt中b是P的右子集隶属度）
	def deblur(self, l_sub_Y, r_sub_Y, l_sub_P, r_sub_P, a, b):
		l_up_sub = self.matrix_blur[l_sub_P][l_sub_Y]
		r_up_sub = self.matrix_blur[l_sub_P][r_sub_Y]
		l_down_sub = self.matrix_blur[r_sub_P][l_sub_Y]
		r_down_sub = self.matrix_blur[r_sub_P][r_sub_Y]

		l_up_Hp = self.map_deblur_Hp[l_up_sub]
		r_up_Hp = self.map_deblur_Hp[r_up_sub]
		l_down_Hp = self.map_deblur_Hp[l_down_sub]
		r_down_Hp = self.map_deblur_Hp[r_down_sub]

		l_up_Q = self.map_deblur_Q[l_up_sub]
		r_up_Q = self.map_deblur_Q[r_up_sub]
		l_down_Q = self.map_deblur_Q[l_down_sub]
		r_down_Q = self.map_deblur_Q[r_down_sub]

		Hp = a*b*l_up_Hp + (1-a)*b*r_up_Hp + a*(1-b)*l_down_Hp + (1-a)*(1-b)*r_down_Hp
		Q = a*b*l_up_Q + (1-a)*b*r_up_Q + a*(1-b)*l_down_Q + (1-a)*(1-b)*r_down_Q
		return Hp, Q

	#即ppt中模糊化所示的△Y和p
	def get_Hp_Q(self, Y, P):
		l_sub_Y, r_sub_Y, a, _a = self.blur_Y(Y)
		l_sub_P, r_sub_P, b, _b = self.blur_P(P)
		Hp, Q = self.deblur(l_sub_Y, r_sub_Y, l_sub_P, r_sub_P, a, b)
		return Hp, Q


#blur = Blur()
#print(blur.get_Hp_Q(180, 2))


















