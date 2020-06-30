from InputParam import InputParam
'''
注意这里导入的param是实例，最好别再另外实例化类了（单例模式），
直接调用里面的方法就行，因为随着流程的进行，实例param里的参数可能会变化。
'''
param = InputParam()
a = param.get_range_intake_air()
print("进气量范围: ", a)

a = param.get_range_intake_material()
print("进料量范围: ", a)

a = param.get_range_derta_intake_air()
print("进气量变化范围: ", a)

a = param.get_range_derta_intake_material()
print("进料量变化范围: ", a)

#进气量
intake_air = 13.0
#进料量
intake_material = 30.0
#H2、CH4、CO（0-100的百分数）
H2 = 9.0
CH4 = 3.0
CO = 18.0
#气化温度（摄氏度）
T = 830.0
#进气量变化量
derta_intake_air = 2.7
#进料量变化量
derta_intake_material = 12.0

a = param.get_cost(intake_air, intake_material, H2, CH4, CO, T, derta_intake_air, derta_intake_material)
print("成本: ", a)

#set_start_time（time）可以设置时间点，默认是0时刻的内容（上面都是0时刻的）
#time的单位是min，设置完时间后，控制时域和成本函数都会变化，即先设置时间，再进行运算

param.set_start_time(0)
print('0min时刻：')
a = param.get_control_time()
print("控制时域: ", a)
a = param.get_cost(intake_air, intake_material, H2, CH4, CO, T, derta_intake_air, derta_intake_material)
print("成本: ", a)

param.set_start_time(10)
print('10min时刻：')
a = param.get_control_time()
print("控制时域: ", a)
a = param.get_cost(intake_air, intake_material, H2, CH4, CO, T, derta_intake_air, derta_intake_material)
print("成本: ", a)