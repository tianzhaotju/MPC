import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class MainWindow(QWidget):

	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		#窗口大小和布局
		width = QApplication.desktop().screenGeometry(0).width()
		height = QApplication.desktop().screenGeometry(0).height()
		self.resize(width*5/7,height*5/6)
		self.center()

		#窗口标题
		self.setWindowTitle('气化炉控制系统软件')

		titleFont = QFont("黑体",14,QFont.Bold)
		subTitleFont = QFont("黑体", 12,QFont.Bold)
		textFont = QFont("宋体", 10)

		br = QLabel()

		#1 成本函数1：控制参考曲线
		self.title1 = QLabel("成本函数1：控制参考曲线")
		self.title1.setFont(titleFont)

		#1.1 负荷
		self.checkBox1_1 = QCheckBox()
		self.checkBox1_1.setChecked(True)
		self.text1_1 = QLabel("负荷")
		self.text1_1.setFont(textFont)
		self.input1_1 = QLineEdit()
		self.input1_1.setPlaceholderText("权重(0-1)")
		self.input1_1.setStyleSheet("width:40%;")
		self.button1_1 = QPushButton("导入/修改负荷需求")
		self.button1_1.clicked.connect(self.test)
		hbox1_1 = QHBoxLayout()
		hbox1_1.addWidget(self.checkBox1_1)
		hbox1_1.addWidget(self.text1_1)
		hbox1_1.addWidget(self.input1_1)
		hbox1_1.addWidget(self.button1_1)

		#1.2 热值
		self.checkBox1_2 = QCheckBox()
		self.checkBox1_2.setChecked(True)
		self.text1_2 = QLabel("热值")
		self.text1_2.setFont(textFont)
		self.input1_2 = QLineEdit()
		self.input1_2.setPlaceholderText("权重(0-1)")
		self.input1_2.setStyleSheet("width:40%;")
		self.button1_2 = QPushButton("导入/修改热值需求")
		self.button1_2.clicked.connect(self.test)
		hbox1_2 = QHBoxLayout()
		hbox1_2.addWidget(self.checkBox1_2)
		hbox1_2.addWidget(self.text1_2)
		hbox1_2.addWidget(self.input1_2)
		hbox1_2.addWidget(self.button1_2)

		#1.3 H2%
		self.checkBox1_3 = QCheckBox()
		self.checkBox1_3.setChecked(True)
		self.text1_3 = QLabel("H2%")
		self.text1_3.setFont(textFont)
		self.input1_3 = QLineEdit()
		self.input1_3.setPlaceholderText("权重(0-1)")
		self.input1_3.setStyleSheet("width:40%;")
		self.button1_3 = QPushButton("导入/修改H2%需求")
		self.button1_3.clicked.connect(self.test)
		hbox1_3 = QHBoxLayout()
		hbox1_3.addWidget(self.checkBox1_3)
		hbox1_3.addWidget(self.text1_3)
		hbox1_3.addWidget(self.input1_3)
		hbox1_3.addWidget(self.button1_3)

		#1.4 CO:CO2
		self.checkBox1_4 = QCheckBox()
		self.checkBox1_4.setChecked(True)
		self.text1_4 = QLabel("CO/CO2")
		self.text1_4.setFont(textFont)
		self.input1_4 = QLineEdit()
		self.input1_4.setPlaceholderText("权重(0-1)")
		self.input1_4.setStyleSheet("width:40%;")
		self.button1_4 = QPushButton("导入/修改CO:CO2需求")
		self.button1_4.clicked.connect(self.test)
		hbox1_4 = QHBoxLayout()
		hbox1_4.addWidget(self.checkBox1_4)
		hbox1_4.addWidget(self.text1_4)
		hbox1_4.addWidget(self.input1_4)
		hbox1_4.addWidget(self.button1_4)

		#2 成本函数2：控制量变化幅值
		self.title2 = QLabel("成本函数2：控制量变化幅值")
		self.title2.setFont(titleFont)

		#2.1 进气量相邻时间变化范围
		self.input2_1_1 = QLineEdit()
		self.input2_1_1.setPlaceholderText("权重(0-1)")
		self.input2_1_1.setStyleSheet("width:40%;")
		self.text2_1_1 = QLabel("进气量相邻时间变化范围")
		self.text2_1_1.setFont(textFont)
		self.input2_1_2 = QLineEdit()
		self.input2_1_2.setPlaceholderText("min")
		self.input2_1_2.setStyleSheet("width:40%;")
		self.text2_1_2 = QLabel("——")
		self.text2_1_2.setFont(textFont)
		self.input2_1_3 = QLineEdit()
		self.input2_1_3.setPlaceholderText("max")
		self.input2_1_3.setStyleSheet("width:40%;")
		hbox2_1 = QHBoxLayout()
		hbox2_1.addWidget(self.text2_1_1)
		hbox2_1.addWidget(self.input2_1_2)
		hbox2_1.addWidget(self.text2_1_2)
		hbox2_1.addWidget(self.input2_1_3)
		hbox2_1.addWidget(self.input2_1_1)

		#2.2 进料量相邻时间变化范围
		self.input2_2_1 = QLineEdit()
		self.input2_2_1.setPlaceholderText("权重(0-1)")
		self.input2_2_1.setStyleSheet("width:40%;")
		self.text2_2_1 = QLabel("进料量相邻时间变化范围")
		self.text2_2_1.setFont(textFont)
		self.input2_2_2 = QLineEdit()
		self.input2_2_2.setPlaceholderText("min")
		self.input2_2_2.setStyleSheet("width:40%;")
		self.text2_2_2 = QLabel("——")
		self.text2_2_2.setFont(textFont)
		self.input2_2_3 = QLineEdit()
		self.input2_2_3.setPlaceholderText("max")
		self.input2_2_3.setStyleSheet("width:40%;")
		hbox2_2 = QHBoxLayout()
		hbox2_2.addWidget(self.text2_2_1)
		hbox2_2.addWidget(self.input2_2_2)
		hbox2_2.addWidget(self.text2_2_2)
		hbox2_2.addWidget(self.input2_2_3)
		hbox2_2.addWidget(self.input2_2_1)

		#2.3 进气量范围
		self.text2_3_1 = QLabel("进气量范围")
		self.text2_3_1.setFont(textFont)
		self.input2_3_1 = QLineEdit()
		self.input2_3_1.setPlaceholderText("min")
		self.input2_3_1.setStyleSheet("width:40%;")
		self.text2_3_2 = QLabel("——")
		self.text2_3_2.setFont(textFont)
		self.input2_3_2 = QLineEdit()
		self.input2_3_2.setPlaceholderText("max")
		self.input2_3_2.setStyleSheet("width:40%;")
		hbox2_3 = QHBoxLayout()
		hbox2_3.addWidget(self.text2_3_1)
		hbox2_3.addWidget(self.input2_3_1)
		hbox2_3.addWidget(self.text2_3_2)
		hbox2_3.addWidget(self.input2_3_2)

		#2.4 进料量范围
		self.text2_4_1 = QLabel("进料量范围")
		self.text2_4_1.setFont(textFont)
		self.input2_4_1 = QLineEdit()
		self.input2_4_1.setPlaceholderText("min")
		self.input2_4_1.setStyleSheet("width:40%;")
		self.text2_4_2 = QLabel("——")
		self.text2_4_2.setFont(textFont)
		self.input2_4_2 = QLineEdit()
		self.input2_4_2.setPlaceholderText("max")
		self.input2_4_2.setStyleSheet("width:40%;")
		hbox2_4 = QHBoxLayout()
		hbox2_4.addWidget(self.text2_4_1)
		hbox2_4.addWidget(self.input2_4_1)
		hbox2_4.addWidget(self.text2_4_2)
		hbox2_4.addWidget(self.input2_4_2)

		#3 成本函数3：工艺性能
		self.title3 = QLabel("成本函数3：工艺性能")
		self.title3.setFont(titleFont)

		#3.1 μ1 产气热值
		self.checkBox3_1 = QCheckBox()
		self.checkBox3_1.setChecked(True)
		self.text3_1 = QLabel("产气热值")
		self.text3_1.setFont(textFont)
		self.input3_1 = QLineEdit()
		self.input3_1.setPlaceholderText("权重(0-1)")
		self.input3_1.setStyleSheet("width:40%;")
		hbox3_1 = QHBoxLayout()
		hbox3_1.addWidget(self.checkBox3_1)
		hbox3_1.addWidget(self.text3_1)
		hbox3_1.addWidget(self.input3_1)

		#3.2 μ2 气化温度
		self.checkBox3_2 = QCheckBox()
		self.checkBox3_2.setChecked(True)
		self.text3_2 = QLabel("气化温度")
		self.text3_2.setFont(textFont)
		self.input3_2 = QLineEdit()
		self.input3_2.setPlaceholderText("权重(0-1)")
		self.input3_2.setStyleSheet("width:40%;")
		hbox3_2 = QHBoxLayout()
		hbox3_2.addWidget(self.checkBox3_2)
		hbox3_2.addWidget(self.text3_2)
		hbox3_2.addWidget(self.input3_2)


		#3.3 μ3 气化效率
		self.checkBox3_3 = QCheckBox()
		self.checkBox3_3.setChecked(True)
		self.text3_3 = QLabel("气化效率")
		self.text3_3.setFont(textFont)
		self.input3_3 = QLineEdit()
		self.input3_3.setPlaceholderText("权重(0-1)")
		self.input3_3.setStyleSheet("width:40%;")
		hbox3_3 = QHBoxLayout()
		hbox3_3.addWidget(self.checkBox3_3)
		hbox3_3.addWidget(self.text3_3)
		hbox3_3.addWidget(self.input3_3)

		#4 成本函数间的权重
		self.title4 = QLabel("成本函数间的权重")
		self.title4.setFont(titleFont)

		#4.1
		self.text4_1 = QLabel("成本函数1 : 成本函数2 : 成本函数3 = ")
		self.text4_1.setFont(textFont)
		hbox4_1 = QHBoxLayout()
		hbox4_1.addWidget(self.text4_1)

		#4.2
		self.input4_2_1 = QLineEdit()
		self.input4_2_1.setPlaceholderText("权重(0-1)")
		self.input4_2_1.setStyleSheet("width:40%;")
		self.input4_2_2 = QLineEdit()
		self.input4_2_2.setPlaceholderText("权重(0-1)")
		self.input4_2_2.setStyleSheet("width:40%;")
		self.input4_2_3 = QLineEdit()
		self.input4_2_3.setPlaceholderText("权重(0-1)")
		self.input4_2_3.setStyleSheet("width:40%;")
		hbox4_2 = QHBoxLayout()
		hbox4_2.addWidget(self.input4_2_1)
		hbox4_2.addWidget(QLabel(" : "))
		hbox4_2.addWidget(self.input4_2_2)
		hbox4_2.addWidget(QLabel(" : "))
		hbox4_2.addWidget(self.input4_2_3)

		#5 原料属性
		self.title5 = QLabel("原料特征")
		self.title5.setFont(titleFont)

		#5.1 CH
		self.text5_1_1 = QLabel("C%: ")
		self.text5_1_1.setFont(textFont)
		self.input5_1_1 = QLineEdit()
		self.input5_1_1.setPlaceholderText("0-100")
		self.input5_1_1.setStyleSheet("width:40%;")
		self.text5_1_2 = QLabel("H%: ")
		self.text5_1_2.setFont(textFont)
		self.input5_1_2 = QLineEdit()
		self.input5_1_2.setPlaceholderText("0-100")
		self.input5_1_2.setStyleSheet("width:40%;")
		hbox5_1 = QHBoxLayout()
		hbox5_1.addWidget(self.text5_1_1)
		hbox5_1.addWidget(self.input5_1_1)
		hbox5_1.addWidget(self.text5_1_2)
		hbox5_1.addWidget(self.input5_1_2)

		#5.2 ONS
		self.text5_2_1 = QLabel("O%: ")
		self.text5_2_1.setFont(textFont)
		self.input5_2_1 = QLineEdit()
		self.input5_2_1.setPlaceholderText("0-100")
		self.input5_2_1.setStyleSheet("width:20%;")
		self.text5_2_2 = QLabel("N%: ")
		self.text5_2_2.setFont(textFont)
		self.input5_2_2 = QLineEdit()
		self.input5_2_2.setPlaceholderText("0-100")
		self.input5_2_2.setStyleSheet("width:20%;")
		self.text5_2_3 = QLabel("S%: ")
		self.text5_2_3.setFont(textFont)
		self.input5_2_3 = QLineEdit()
		self.input5_2_3.setPlaceholderText("0-100")
		self.input5_2_3.setStyleSheet("width:20%;")
		hbox5_2 = QHBoxLayout()
		hbox5_2.addWidget(self.text5_2_1)
		hbox5_2.addWidget(self.input5_2_1)
		hbox5_2.addWidget(self.text5_2_2)
		hbox5_2.addWidget(self.input5_2_2)
		hbox5_2.addWidget(self.text5_2_3)
		hbox5_2.addWidget(self.input5_2_3)

		#6 运行过程
		self.title6 = QLabel("运行过程")
		self.title6.setFont(titleFont)
		self.runButton = QPushButton("开始运行")
		self.input6_1 = QLineEdit()
		self.input6_1.setPlaceholderText("优化迭代次数")
		self.input6_1.setStyleSheet("width:20%;")
		hbox6_1 = QHBoxLayout()
		hbox6_1.addWidget(self.runButton)
		hbox6_1.addWidget(self.input6_1)
		self.text6 = QTextEdit()

		#7 运行结果
		self.title7 = QLabel("运行结果")
		self.title7.setFont(titleFont)

		#7.1 控制量
		self.text7_1 = QLabel("控制量：")
		self.text7_1.setFont(textFont)
		self.button7_1_1 = QPushButton("查看结果")
		self.button7_1_1.clicked.connect(self.test)
		self.button7_1_2 = QPushButton("导出数据")
		self.button7_1_2.clicked.connect(self.test)
		hbox7_1 = QHBoxLayout()
		hbox7_1.addWidget(self.text7_1)
		hbox7_1.addWidget(self.button7_1_1)
		hbox7_1.addWidget(self.button7_1_2)

		#7.2 参考曲线
		self.text7_2 = QLabel("参考曲线：")
		self.text7_2.setFont(textFont)
		self.button7_2_1 = QPushButton("查看结果")
		self.button7_2_1.clicked.connect(self.test)
		self.button7_2_2 = QPushButton("导出数据")
		self.button7_2_2.clicked.connect(self.test)
		hbox7_2 = QHBoxLayout()
		hbox7_2.addWidget(self.text7_2)
		hbox7_2.addWidget(self.button7_2_1)
		hbox7_2.addWidget(self.button7_2_2)

		#7.3 工艺性能
		self.text7_3 = QLabel("工艺性能：")
		self.text7_3.setFont(textFont)
		self.button7_3_1 = QPushButton("查看结果")
		self.button7_3_1.clicked.connect(self.test)
		self.button7_3_2 = QPushButton("导出数据")
		self.button7_3_2.clicked.connect(self.test)
		hbox7_3 = QHBoxLayout()
		hbox7_3.addWidget(self.text7_3)
		hbox7_3.addWidget(self.button7_3_1)
		hbox7_3.addWidget(self.button7_3_2)

		mainLayout = QGridLayout()
		#1 成本函数1：控制参考曲线
		mainLayout.addWidget(br, 0, 0)
		mainLayout.addWidget(self.title1, 		1, 0, 1, 5, Qt.AlignLeft|Qt.AlignBottom)
		mainLayout.addLayout(hbox1_1, 2, 0, 1, 3)
		mainLayout.addLayout(hbox1_2, 3, 0, 1, 3)
		mainLayout.addLayout(hbox1_3, 4, 0, 1, 3)
		mainLayout.addLayout(hbox1_4, 5, 0, 1, 3)

		#2 成本函数2：控制量变化幅值
		mainLayout.addWidget(self.title2, 6, 0, 1, 5, Qt.AlignLeft|Qt.AlignHCenter)
		mainLayout.addLayout(hbox2_1, 7, 0, 1, 3)
		mainLayout.addLayout(hbox2_2, 8, 0, 1, 3)
		mainLayout.addLayout(hbox2_3, 9, 0, 1, 3)
		mainLayout.addLayout(hbox2_4, 10, 0, 1, 3)
		#mainLayout.addWidget(br, 12, 0)

		#3 成本函数3：工艺性能
		mainLayout.addWidget(self.title3, 11, 0, 1, 5, Qt.AlignLeft|Qt.AlignHCenter)
		mainLayout.addLayout(hbox3_1, 12, 0, 1, 2)
		mainLayout.addLayout(hbox3_2, 13, 0, 1, 2)
		mainLayout.addLayout(hbox3_3, 14, 0, 1, 2)
		#mainLayout.addWidget(br, 17, 0)

		#4 成本函数间的权重
		mainLayout.addWidget(self.title4, 15, 0, 1, 5, Qt.AlignLeft|Qt.AlignHCenter)
		mainLayout.addLayout(hbox4_1, 16, 0, 1, 3)
		mainLayout.addLayout(hbox4_2, 17, 0, 1, 3)
		#mainLayout.addWidget(br, 21, 0)

		#5 原料特征
		mainLayout.addWidget(self.title5, 18, 0, 1, 5, Qt.AlignLeft|Qt.AlignBottom)
		mainLayout.addLayout(hbox5_1, 19, 0, 1, 2)
		mainLayout.addLayout(hbox5_2, 20, 0, 1, 2)

		#6 运行过程
		mainLayout.addWidget(self.title6, 		1, 4, 1, 1, Qt.AlignLeft|Qt.AlignTop)
		mainLayout.addLayout(hbox6_1,		 	2, 4, 1, 2, Qt.AlignLeft|Qt.AlignBottom)
		mainLayout.addWidget(self.text6, 		3, 4, 12, 5, Qt.AlignLeft|Qt.AlignHCenter)

		#7 运行结果
		mainLayout.addWidget(self.title7, 15, 4, 1, 1, Qt.AlignLeft|Qt.AlignBottom)
		mainLayout.addLayout(hbox7_1, 16, 4, 1, 2)
		mainLayout.addLayout(hbox7_2, 17, 4, 1, 2)
		mainLayout.addLayout(hbox7_3, 18, 4, 1, 2)

		self.setLayout(mainLayout)
		self.show()

	#主窗口居中
	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	#调色板调用
	def colorSelect1(self):
		col = QColorDialog.getColor()
		if col.isValid():
			self.changeColor(col.name())

	#常见材质颜色选择
	def colorSelect2(self):
		colorDialog = MaterialColorDialog()
		if colorDialog.exec_() == QDialog.Accepted:
			pass
		if colorDialog.colorIndex != -1:
			self.changeColor(colorDialog.result)

	def changeColor(self, newColor):
		self.colorLabel.setStyleSheet('QWidget {background-color:%s}' % newColor)
		self.selectedColor = newColor
		h, s, v = hex_hsv(newColor)
		self.colorLabel.setToolTip('<b>HSB:</b> ('+ str(h) + ', ' + str(s) +'%, ' + str(v) +'%)')

	#输入滑块名字和序号，获得一组横向滑块
	def getSplider(self, name, index):
		#创建滑块
		splider = QSlider(Qt.Horizontal, self)
		splider.setMinimum(0)
		splider.setMaximum(100)
		splider.setTickInterval(10)
		splider.valueChanged.connect(lambda: self.changeValue(splider, value, index))

		#创建文本框
		label = QLabel()
		label.setFont(QFont("宋体",10))
		label.setText(name)
		value = QLabel()
		value.setFont(QFont("宋体",10))
		value.setNum(0)

		#复选框
		checkBox = QCheckBox()
		checkBox.setChecked(True)
		checkBox.stateChanged.connect(lambda:self.whichCanUse(label, value, splider, index))
		checkBox.stateChanged.connect(self.canUse)
		checkBox.setStyleSheet("""
QCheckBox::indicator{
					width: 25px;
					height: 25px;
				}""")
		#横向Layout
		hbox = QHBoxLayout()
		subHBox = QHBoxLayout()
		#添加靠左对齐的控件
		subHBox.addWidget(checkBox, Qt.AlignLeft)
		subHBox.addWidget(label, Qt.AlignLeft)
		subHBox.addWidget(value, Qt.AlignLeft)
		hbox.addLayout(subHBox)
		hbox.addWidget(splider, Qt.AlignLeft)
		#设置0,1列的比例为1:3
		hbox.setStretch(0, 1)
		hbox.setStretch(1, 3)

		return hbox

	#滑块取值事件
	def changeValue(self, splider, value, index):
		value.setNum(splider.value())
		self.splidersValue[index] = splider.value()
		#print(self.splidersValue[index])

	def whichCanUse(self, label, value, splider, index):
		self.tempIndex = index
		self.tempLabel = label
		self.tempValue = value
		self.tempSplider = splider
		#穿孔
		#if index == 4:
			#print("穿孔")

	#复选框检查
	def canUse(self, state):
		if state != Qt.Checked:
			self.tempLabel.setEnabled(False)
			self.tempValue.setEnabled(False)
			self.tempSplider.setEnabled(False)
			self.splidersValue[self.tempIndex] = -1
		else:
			self.tempLabel.setEnabled(True)
			self.tempValue.setEnabled(True)
			self.tempSplider.setEnabled(True)
			self.splidersValue[self.tempIndex] = self.tempSplider.value()
		if self.tempIndex == 4:
			if state != Qt.Checked:
				self.text4.setVisible(False)
				self.chuanKongRadio1.setVisible(False)
				self.chuanKongRadio2.setVisible(False)
				self.chuanKongRadio3.setVisible(False)
			else:
				self.text4.setVisible(True)
				self.chuanKongRadio1.setVisible(True)
				self.chuanKongRadio2.setVisible(True)
				self.chuanKongRadio3.setVisible(True)


	def radioBtnClicked(self):
		self.colorType = self.radioGroup.checkedId()
		#print(self.colorType)

	def chuanKong1Checked(self, state):
		if state != Qt.Checked:
			self.chuanKongType[0] = 0
		else:
			self.chuanKongType[0] = 1

	def chuanKong2Checked(self, state):
		if state != Qt.Checked:
			self.chuanKongType[1] = 0
		else:
			self.chuanKongType[1] = 1

	def chuanKong3Checked(self, state):
		if state != Qt.Checked:
			self.chuanKongType[2] = 0
		else:
			self.chuanKongType[2] = 1

	def search(self):
		mColor = self.selectedColor
		mColorType = self.colorType
		mValue0 = self.splidersValue[0]
		mValue1 = self.splidersValue[1]
		mValue2 = self.splidersValue[2]
		mValue3 = self.splidersValue[3]
		mValue4 = self.splidersValue[4]
		chuanKongType = self.chuanKongType
		excel = DBExcel('data/', mColor, mColorType, mValue0, mValue1, mValue2, mValue3, mValue4, chuanKongType)
		materials = excel.getSelectedMaterial()
		print(len(materials), mColor, mColorType, mValue0, mValue1, mValue2, mValue3, mValue4, chuanKongType)
		if len(materials) == 0:
			reply = QMessageBox.information(self,
										"无结果",  
										"没有根据您的条件匹配到合适的材料。",  
										QMessageBox.Ok)
			return
		resultDialog = ResultDialog(materials)
		if resultDialog.exec_() == QDialog.Accepted:
			pass

	def test(self):
		return






