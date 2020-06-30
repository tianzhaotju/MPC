#demo_14:关于TableWidGet的使用，注意：在table表头分为水平和垂直两种，及horizontal header和vertical header两类。
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from ShowPicture import ShowPicture

initArgs = [
			["09:00", 100],
			["10:00", 120],
			["11:00", 130],
			]

class ResultDialog(QDialog):

	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setWindowTitle('负荷需求')
		self.center()
		self.args = initArgs
		self.newArgs = initArgs

		#初始化表格模块
		self.initTable()
		#初始化按钮模块
		self.initBtn()

		#table布局
		tableLayout = QVBoxLayout()
		tableLayout.addWidget(self.table)
		#按钮布局
		buttonLayout = QHBoxLayout()
		buttonLayout.addWidget(self.okBtn)
		buttonLayout.addWidget(self.cancelBtn)

		#总体布局
		mainLayout = QVBoxLayout()
		mainLayout.addLayout(tableLayout)
		mainLayout.addLayout(buttonLayout)

		self.setLayout(mainLayout)
		self.show()

	def center(self):
		#调整大小
		width = QApplication.desktop().screenGeometry(0).width()
		height = QApplication.desktop().screenGeometry(0).height()
		self.resize(width/3,height*2/3)
		#中心
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def initTable(self):
		titles = ['时间', '负荷（kW）']

		self.table = QTableWidget()
		self.table.setRowCount(20)
		self.table.setColumnCount(2)
		self.table.setHorizontalHeaderLabels(titles)

		#表格行
		#self.table.horizontalHeader().setStyleSheet("background-color: blue");
		#self.table.setEditTriggers(QTableWidget.NoEditTriggers)#单元格不可编辑
		self.table.setSelectionBehavior(QTableWidget.SelectRows)  #选中列还是行，这里设置选中行
		self.table.setSelectionMode(QTableWidget.SingleSelection) #只能选中一行或者一列
		self.table.horizontalHeader().setStretchLastSection(True)  #列宽度占满表格(最后一个列拉伸处理沾满表格)
		self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);#所有列自适应表格宽度

		#1、设置每一个标题单元格样式
		# for i in range(self.table.columnCount()):
		#       headItem = self.table.horizontalHeaderItem(i)
		#       headItem.setFont(QFont("song", 14, QFont.Bold))
		#       headItem.setForeground(QBrush(Qt.gray))
		#       headItem.setBackgroundColor(QColor(0, 60, 10))      # 设置单元格背景颜色
		#       #headItem.setTextColor(QColor(200, 111, 30))        # 设置文字颜色
		#       headItem.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
		#2、设置整个表格列标题样式
		font = self.table.horizontalHeader().font()
		font.setBold(True)
		self.table.horizontalHeader().setFont(font)
		#self.table.setFrameShape(QFrame.NoFrame)                   #设置表格外层无边框
		#self.table.setShowGrid(False)                              #是否显示单元格网格线False 则不显示
		#self.table.horizontalHeader().setHighlightSections(False)  #设置表格列头不塌陷
		#self.table.horizontalHeader().setFixedHeight(35)           #设置表列头高度
		#self.table.verticalHeader().setVisible(False)            #设置隐藏行头
		#self.table.horizontalHeader().setFixedWidth(820)           #设置列标题头所在行，宽度（没啥用）

		#设置表格的滚动调样式：self.table.horizontalScrollBar().setStyleSheet.... ,窗体的也可以设置：self.horizontalScrollBar().setStyleSheet...
		self.table.horizontalScrollBar().setStyleSheet("QScrollBar{background:transparent; height:10px;}"
											"QScrollBar::handle{background:lightgray; border:2px solid transparent; border-radius:5px;}"
											"QScrollBar::handle:hover{background:gray;}"
											"QScrollBar::sub-line{background:transparent;}"
											"QScrollBar::add-line{background:transparent;}");
		self.table.verticalScrollBar().setStyleSheet("QScrollBar{background:transparent; width: 10px;}"
											"QScrollBar::handle{background:lightgray; border:2px solid transparent; border-radius:5px;}"
											"QScrollBar::handle:hover{background:gray;}"
											"QScrollBar::sub-line{background:transparent;}"
											"QScrollBar::add-line{background:transparent;}");
		#设置选中行样式
		self.table.setStyleSheet("selection-background-color: #BBBBBB");

		row_count = self.table.rowCount()
		self.table.insertRow(row_count)

	def initBtn(self):
		self.okBtn = QPushButton('确认', self)
		self.cancelBtn = QPushButton('取消', self)

		self.okBtn.clicked.connect(self.close)
		self.cancelBtn.clicked.connect(self.close)






