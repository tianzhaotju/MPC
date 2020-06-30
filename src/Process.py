import time

class TotalFrame:

	def __init__(self):
		#过程开始时间
		self.__startTime = time.time()
		#预测间隔
		self.predictCycle = 10 #10s
		#上次预测时间
		self.lastPredictTime = -self.predictCycle

	#整个过程默认开始时间为0
	def getNow(self):
		return time.time() - self.__startTime

	def getTimeByFloat(self, floatTime):
		return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(floatTime))

	def run(self):
		while True:
			now = self.getNow()
			if now - self.lastPredictTime >= self.predictCycle:
				print("开始预测：", self.getTimeByFloat(now))
				self.lastPredictTime = now
			time.sleep(5)

	def test(self):
		first = time.time()
		time.sleep(2)
		last = time.time()
		derta = last - first;
		print(first)
		print(last)
		print(derta)


p = TotalFrame()
p.run()