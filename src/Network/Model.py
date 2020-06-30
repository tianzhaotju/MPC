import torch
import os
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from LoadData import DataSet
import time
from Net import Net
from LSTM import LSTM


class Model():

    def __init__(self, stride =5, input_size=6, output_size=4,learning_rate=0.00001,EPOCH = 1000):
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.EPOCH = EPOCH
        self.X_MIN = [0, 9.5]
        self.X_MAX = [100, 16.5]
        self.Y_MIN = [360, 4, 1, 10]
        self.Y_MAX = [850, 12, 4, 28]

    # Traverses the specified directory to display all filenames under the directory.
    def eachFile(self,filepath):
        result = []
        pathDir = os.listdir(filepath)
        for allDir in pathDir:
            child = os.path.join('%s%s' % (filepath, allDir))
            result.append(child)
        return result

    # Read the contents of the file and save it.
    def readFile(self,filename,x,y):
        fopen = open(filename, 'r')
        for eachLine in fopen:
            eachLineData = eachLine.split(",")
            #Take out the x and y data
            x_line = []
            for i in range(0,len(eachLineData)-self.output_size):
                x_line.append(float(eachLineData[i]))
            x.append(np.array(x_line))

            y_line = []
            for i in range(len(eachLineData)-self.output_size, len(eachLineData)):
                y_line.append(float(eachLineData[i]))
            y.append(np.array(y_line))
        fopen.close()
        return x,y

    # 获取数据
    def getData(self,data_files_dic,normalized,percent):
        data_files_path = self.eachFile(data_files_dic)

        data_x = []
        data_y = []
        for data_file in data_files_path:
            data_x,data_y = self.readFile(data_file,data_x,data_y)
        data_x = np.array(data_x)[:,1:]
        data_y = np.array(data_y)
        if normalized:
            data_x,data_y = self.normalizedData(data_x,data_y)

        test_data_len = int((1-percent)*len(data_x))
        # np.random.seed(0)
        # state = np.random.get_state()
        # np.random.shuffle(data_x)
        # np.random.set_state(state)
        # np.random.shuffle(data_y)

        train_data_x = data_x[test_data_len:]
        train_data_y = data_y[test_data_len:]
        test_data_x = data_x[:test_data_len]
        test_data_y = data_y[:test_data_len]


        train_data_x_ = []
        train_data_y_ = []
        test_data_x_ = []
        test_data_y_ = []

        for i in range(len(train_data_x)-self.stride):
            temp_input = np.array(train_data_x[i:i+self.stride,0:])
            temp_output = np.array(train_data_y[i:i + self.stride, 0:])
            temp = np.concatenate((temp_input,temp_output),axis=1)
            temp = temp.flatten()
            train_data_x_.append(temp)
            temp = train_data_y[i+self.stride]
            train_data_y_.append(temp)

        for i in range(len(test_data_x)-self.stride):
            temp_input = np.array(test_data_x[i:i + self.stride, 0:])
            temp_output = np.array(test_data_y[i:i + self.stride, 0:])
            temp = np.concatenate((temp_input, temp_output), axis=1)
            temp = temp.flatten()
            test_data_x_.append(temp)
            temp = test_data_y[i+self.stride]
            test_data_y_.append(temp)

        train_data_x_ = np.array(train_data_x_)
        train_data_y_ = np.array(train_data_y_)
        test_data_x_ = np.array(test_data_x_)
        test_data_y_ = np.array(test_data_y_)

        return train_data_x_,train_data_y_,test_data_x_,test_data_y_

    def getOnlineData(self,data_files_dic,normalized):
        data_x = []
        data_y = []
        data_x,data_y = self.readFile(data_files_dic,data_x,data_y)
        data_x = np.array(data_x)[:,1:]
        data_y = np.array(data_y)
        if normalized:
            data_x,data_y = self.normalizedData(data_x,data_y)

        data_x_ = []
        data_y_ = []

        for i in range(len(data_x)-self.stride):
            temp_input = np.array(data_x[i:i+self.stride,0:])
            temp_output = np.array(data_y[i:i + self.stride, 0:])
            temp = np.concatenate((temp_input,temp_output),axis=1)
            temp = temp.flatten()
            data_x_.append(temp)
            temp = data_y[i+self.stride]
            data_y_.append(temp)

        data_x_ = np.array(data_x_)
        data_y_ = np.array(data_y_)

        return data_x_, data_y_

    #Normalized data
    def normalizedData(self,data_x,data_y):
        for i in range(data_x.shape[1]):
            # max_data_x = np.max(data_x[:, i])
            # data_x[:, i]/=max_data_x
            data_x[:, i] = (data_x[:, i]-self.X_MIN[i])/(self.X_MAX[i]-self.X_MIN[i])
        for i in range(data_y.shape[1]):
            # max_data_y = np.max(data_y[:, i])
            # data_y[:, i] /= max_data_y
            data_y[:, i] = (data_y[:, i] - self.Y_MIN[i]) / (self.Y_MAX[i] - self.Y_MIN[i])

        return data_x, data_y

    # Normalized data X
    def normalizedDataX(self, data_x):
        data_x = np.array(data_x)
        print(data_x.shape)
        for i in range(2):
            # max_data_x = np.max(data_x[:, i])
            # data_x[:, i]/=max_data_x
            data_x[:, i] = (data_x[:, i] - self.X_MIN[i]) / (self.X_MAX[i] - self.X_MIN[i])

        for i in range(4):
            # max_data_x = np.max(data_x[:, i])
            # data_x[:, i]/=max_data_x
            data_x[:, 2+i] = (data_x[:, 2+i] - self.Y_MIN[i]) / (self.Y_MAX[i] - self.Y_MIN[i])

        return data_x

    # De-Normalized data Y
    def de_normalizedDataY(self,data_y):
        data_y = np.array(data_y)
        for i in range(data_y.shape[1]):
            # max_data_y = np.max(data_y[:, i])
            # data_y[:, i] /= max_data_y
            data_y[:, i] = data_y[:, i] *(self.Y_MAX[i] - self.Y_MIN[i]) + self.Y_MIN[i]

        return data_y

    def train_Net(self,data_files_dic="./data/",normalized=False,percent=0.9):
        f = open("./log/train_Net_"+str(self.stride)+".txt","w")
        print("-------------------------------------------")
        f.write("\n"+"-------------------------------------------")
        # print("Train Net Model...")
        f.write("\n"+"Train Net Model...")
        # all trian data and test data files path

        # Get train data and test data
        train_data_x, train_data_y, test_data_x, test_data_y = self.getData(data_files_dic=data_files_dic, normalized=normalized,
                                                                       percent=percent)

        # Print the size of train data and test data
        print("Train data shape:")
        f.write("\n"+"Train data shape:")
        print(train_data_x.shape)
        f.write("\n"+str(train_data_x.shape))
        print(train_data_y.shape)
        f.write("\n"+str(train_data_y.shape))
        train_data_x = torch.Tensor(train_data_x)
        train_data_y = torch.Tensor(train_data_y)

        trainset = DataSet(train_data_x, train_data_y)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=False)

        # Load the Net LSTM_model
        # net = torch.load('LSTM_model.pth')

        # Bulid the Net_model
        net = Net(self.input_size*self.stride, self.output_size)
        # Optimize all net parameters
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        # The loss function uses mean square error (MSE) loss function
        loss_func = nn.MSELoss()

        for step in range(self.EPOCH):
            total_loss = 0
            for tx, ty in trainloader:
                output = net.forward(torch.unsqueeze(tx, dim=0))
                loss = loss_func(torch.squeeze(output), ty)
                # clear gradients for this training step
                optimizer.zero_grad()
                # back propagation, compute gradients
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            # print(step, float(total_loss))
            f.write("\n"+str(step)+" "+str(float(total_loss)))
        torch.save(net, "./models/Net_model_"+str(self.stride)+".pth")
        f.write("\n"+"Save Net Model")
        print("Save Net Model")

    def test_Net(self,data_files_dic = "./data/", normalized=False, percent=0.9):
        f = open("./log/test_Net_"+str(self.stride)+".txt","w")
        print("-------------------------------------------")
        f.write("\n"+"-------------------------------------------")
        print("Test Net Model...")
        f.write("\n"+"Test Net Model...")
        # all trian data and test data files path

        # Get train data and test data
        train_data_x, train_data_y, test_data_x, test_data_y = self.getData(data_files_dic=data_files_dic, normalized=normalized,
                                                                       percent=percent)

        # Print the size of train data and test data
        print("Test data shape:")
        f.write("\n"+"Test data shape:")
        print(test_data_x.shape)
        f.write("\n"+str(test_data_x))
        print(test_data_y.shape)
        f.write("\n"+str(test_data_y.shape))

        test_data_x = torch.Tensor(test_data_x)
        test_data_y = torch.Tensor(test_data_y)

        testset = DataSet(test_data_x, test_data_y)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        # Load the Net_model
        net = torch.load("./models/Net_model_"+str(self.stride)+".pth")

        loss = np.array([0, 0, 0, 0], float)
        time1 = time.time()
        for tx, ty in testloader:
            output = net(torch.unsqueeze(tx, dim=0))
            output = torch.reshape(output, [output.shape[1], output.shape[2]])
            loss_ = torch.abs(output - ty) / ty
            loss_ = loss_.detach().numpy()[0]
            for i in range(len(loss_)):
                loss[i] = loss[i] + loss_[i]
            print(output.detach().numpy(), ty.numpy())
            f.write("\n"+str(output.detach().numpy())+" "+str(ty.numpy()))
        time2 = time.time()
        loss /= len(testloader)
        print("Net Model Test:")
        f.write("\n"+"Net Model Test:")
        print("---------------------------------------------------")
        f.write("\n"+"---------------------------------------------------")
        print("气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        f.write("\n"+"气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        print("H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        f.write("\n"+"H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        print("CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        f.write("\n"+"CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        print("CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        f.write("\n"+"CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        print("平均耗时：", (time2 - time1) / len(testloader))
        f.write("\n"+"平均耗时："+str((time2 - time1) / len(testloader)))

    def train_LSTM(self,data_files_dic = "./data/", normalized=False, percent=0.9):
        f = open("./log/train_LSTM_"+str(self.stride)+".txt","w")
        print("-------------------------------------------")
        f.write("\n"+"-------------------------------------------")
        print("Train LSTM Model...")
        f.write("\n"+"Train LSTM Model...")
        # all trian data and test data files path

        # Get train data and test data
        train_data_x, train_data_y, test_data_x, test_data_y = self.getData(data_files_dic=data_files_dic,
                                                                            normalized=normalized,
                                                                            percent=percent)

        # Print the size of train data and test data
        print("Train data shape:")
        f.write("\n"+"Train data shape:")
        print(train_data_x.shape)
        f.write("\n"+str(train_data_x.shape))
        print(train_data_y.shape)
        f.write("\n"+str(train_data_y.shape))

        train_data_x = torch.Tensor(train_data_x)
        train_data_y = torch.Tensor(train_data_y)

        trainset = DataSet(train_data_x, train_data_y)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=False)

        # Load the Net LSTM_model
        # net = torch.load('LSTM_model.pth')

        # Bulid the LSTM_model
        net = LSTM(self.input_size*self.stride, self.output_size)
        # Optimize all net parameters
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        # The loss function uses mean square error (MSE) loss function
        loss_func = nn.MSELoss()

        for step in range(self.EPOCH):
            total_loss = 0
            for tx, ty in trainloader:
                output = net.forward(torch.unsqueeze(tx, dim=0))
                # print(output.detach().numpy()[0][0],ty.numpy()[0])
                loss = loss_func(torch.squeeze(output), ty)
                # clear gradients for this training step
                optimizer.zero_grad()
                # back propagation, compute gradients
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            # print(step, float(total_loss))
            f.write("\n"+str(step)+" "+str(float(total_loss)))
        torch.save(net, "./models/LSTM_model_"+str(self.stride)+".pth")
        f.write("\n"+"Save LSTM Model")
        print("Save LSTM Model")

    def test_LSTM(self,data_files_dic = "./data/", normalized=False, percent=0.9):
        f = open("./log/test_LSTM_"+str(self.stride)+".txt","w")
        print("-------------------------------------------")
        f.write("\n"+"-------------------------------------------")
        print("Test LSTM Model...")
        f.write("\n"+"Test LSTM Model...")
        # all trian data and test data files path

        # Get train data and test data
        train_data_x, train_data_y, test_data_x, test_data_y = self.getData(data_files_dic=data_files_dic,
                                                                            normalized=normalized,
                                                                            percent=percent)

        # Print the size of train data and test data
        print("Test data shape:")
        f.write("\n"+"Test data shape:")
        print(test_data_x.shape)
        f.write("\n"+str(test_data_x.shape))
        print(test_data_y.shape)
        f.write("\n"+str(test_data_y.shape))

        test_data_x = torch.Tensor(test_data_x)
        test_data_y = torch.Tensor(test_data_y)

        testset = DataSet(test_data_x, test_data_y)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        # Load the LSTM_model
        print("Load LSTM Model...")
        f.write("\n"+"Load LSTM Model...")
        net = torch.load("./models/LSTM_model_"+str(self.stride)+".pth")

        loss = np.array([0, 0, 0, 0], float)
        time1 = time.time()
        for tx, ty in testloader:
            output = net(torch.unsqueeze(tx, dim=0))
            output = torch.reshape(output, [output.shape[1], output.shape[2]])
            loss_ = torch.abs(output - ty) / ty
            loss_ = loss_.detach().numpy()[0]
            for i in range(len(loss_)):
                loss[i] = loss[i] + loss_[i]
            print(output.detach().numpy(), ty.numpy())
            f.write("\n"+str(output.detach().numpy())+" "+str(ty.numpy()))
        time2 = time.time()
        loss /= len(testloader)
        print("LSTM Model Test:")
        f.write("\n"+"LSTM Model Test:")
        print("---------------------------------------------------")
        f.write("\n"+"---------------------------------------------------")
        print("气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        f.write("\n"+"气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        print("H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        f.write("\n"+"H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        print("CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        f.write("\n"+"CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        print("CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        f.write("\n"+"CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        print("平均耗时：", (time2 - time1) / len(testloader))
        f.write("\n"+"平均耗时："+str((time2 - time1) / len(testloader)))

    def predict_LSTM(self, data_x, normalized=True, de_normalized=True):
        # print("-------------------------------------------")
        if normalized:
            data_x = self.normalizedDataX(data_x)

        data_x = np.array(data_x).flatten()
        data_x = data_x.reshape([1, data_x.shape[0]])
        data_x = torch.Tensor(data_x)

        # Load the Net_model
        net = torch.load("./models/LSTM_model_"+str(self.stride)+".pth")

        data_y = []
        time1 = time.time()
        for tx in data_x:
            tx = torch.reshape(tx,[1,tx.shape[0]])
            output = net(torch.unsqueeze(tx, dim=0))
            output = output.detach().numpy()[0][0]
            time2 = time.time()
            data_y.append(output)
        # print("用时："+str(time2-time1))
        data_y = np.array(data_y)
        if de_normalized:
            data_y = self.de_normalizedDataY(data_y)
        return data_y

    def predict_Net(self, data_x, normalized=True,de_normalized=True):
        # print("-------------------------------------------")
        if normalized:
            data_x = self.normalizedDataX(data_x)

        data_x = np.array(data_x).flatten()
        data_x = data_x.reshape([1,data_x.shape[0]])
        data_x = torch.Tensor(data_x)

        # Load the Net_model
        net = torch.load("./models/Net_model_"+str(self.stride)+".pth")

        data_y = []
        time1 = time.time()
        for tx in data_x:
            output = net(torch.unsqueeze(tx, dim=0))
            output = output.detach().numpy()[0]
            time2 = time.time()
            data_y.append(output)
        # print("用时："+str(time2-time1))
        data_y = np.array(data_y)
        if de_normalized:
            data_y = self.de_normalizedDataY(data_y)
        return data_y

    def test_future_LSTM(self, data_files_dic="./data/", normalized=False, percent=0.9, future=10):
        f = open("./log/test_future_LSTM_s" + str(self.stride)+'_f'+str(future) + ".txt", "w")
        print("-------------------------------------------")
        f.write("\n" + "-------------------------------------------")
        print("Future test LSTM Model...")
        f.write("\n" + "Future test LSTM Model...")
        # all trian data and test data files path
        data_files_path = self.eachFile(data_files_dic)

        # Get test data
        data_x = []
        data_y = []
        for data_file in data_files_path:
            data_x, data_y = self.readFile(data_file, data_x, data_y)
        data_x = np.array(data_x)[:, 1:]
        data_y = np.array(data_y)
        if normalized:
            data_x, data_y = self.normalizedData(data_x, data_y)

        test_data_len = int((1 - percent) * len(data_x))
        test_data_x = data_x[:test_data_len]
        test_data_y = data_y[:test_data_len]
        test_data_x = np.array(test_data_x)
        test_data_y = np.array(test_data_y)

        # Print the size of train data and test data
        print("Test data shape:")
        f.write("\n" + "Test data shape:")
        print(test_data_x.shape)
        f.write("\n" + str(test_data_x.shape))
        print(test_data_y.shape)
        f.write("\n" + str(test_data_y.shape))

        loss = np.array([0, 0, 0, 0], float)
        time1 = time.time()
        for i in range(self.stride,len(test_data_y)-future+1):
            # 需要不断的预测更新
            tx_y = np.array(test_data_y[i - self.stride:i])
            for j in range(i,i+future):
                tx_x = np.array(test_data_x[j - self.stride:j])
                # 更新 tx_y
                if j != i:
                    tx_y = tx_y[1:]
                    tx_y = np.append(tx_y, predict_y, axis=0)
                tx = np.concatenate((tx_x,tx_y),axis=1)
                ty = np.array(test_data_y[j])
                predict_y = self.predict_LSTM(tx, normalized=False,de_normalized=False)
            predict_y = predict_y[0]
            loss_ = np.abs(predict_y - ty) / ty
            for i in range(len(loss_)):
                loss[i] = loss[i] + loss_[i]
            # print(predict_y, ty)
            f.write("\n" + str(predict_y) + " " + str(ty))
        time2 = time.time()
        loss /= (len(test_data_x)-self.stride-future+1)
        print("LSTM Model Test:")
        f.write("\n" + "LSTM Model Test:")
        print("---------------------------------------------------")
        f.write("\n" + "---------------------------------------------------")
        print("气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        f.write("\n" + "气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        print("H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        f.write("\n" + "H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        print("CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        f.write("\n" + "CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        print("CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        f.write("\n" + "CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        print("平均耗时：", (time2 - time1) / (len(test_data_x)-self.stride-future+1))
        f.write("\n" + "平均耗时：" + str((time2 - time1) / (len(test_data_x)-self.stride-future+1)))
        return [str(float(loss[0] * 100))[:8] + "%", str(float(loss[1] * 100))[:8] + "%",
                str(float(loss[2] * 100))[:8] + "%", str(float(loss[3] * 100))[:8] + "%",
                str((time2 - time1) / (len(test_data_x) - self.stride - future + 1))]

    def test_future_Net(self, data_files_dic="./data/", normalized=False, percent=0.9, future=10):
        f = open("./log/test_future_Net_s" + str(self.stride)+'_f'+str(future) + ".txt", "w")
        print("-------------------------------------------")
        f.write("\n" + "-------------------------------------------")
        print("Future test Net Model...")
        f.write("\n" + "Future test Net Model...")
        # all trian data and test data files path
        data_files_path = self.eachFile(data_files_dic)

        # Get test data
        data_x = []
        data_y = []
        for data_file in data_files_path:
            data_x, data_y = self.readFile(data_file, data_x, data_y)
        data_x = np.array(data_x)[:, 1:]
        data_y = np.array(data_y)
        if normalized:
            data_x, data_y = self.normalizedData(data_x, data_y)

        test_data_len = int((1 - percent) * len(data_x))
        test_data_x = data_x[:test_data_len]
        test_data_y = data_y[:test_data_len]
        test_data_x = np.array(test_data_x)
        test_data_y = np.array(test_data_y)

        # Print the size of train data and test data
        print("Test data shape:")
        f.write("\n" + "Test data shape:")
        print(test_data_x.shape)
        f.write("\n" + str(test_data_x.shape))
        print(test_data_y.shape)
        f.write("\n" + str(test_data_y.shape))

        loss = np.array([0, 0, 0, 0], float)
        time1 = time.time()
        for i in range(self.stride,len(test_data_y)-future+1):
            # 需要不断的预测更新
            tx_y = np.array(test_data_y[i - self.stride:i])
            for j in range(i,i+future):
                tx_x = np.array(test_data_x[j - self.stride:j])
                # 更新 tx_y
                if j != i:
                    tx_y = tx_y[1:]
                    tx_y = np.append(tx_y, predict_y, axis=0)
                tx = np.concatenate((tx_x,tx_y),axis=1)
                ty = np.array(test_data_y[j])
                predict_y = self.predict_Net(tx, normalized=False,de_normalized=False)
            predict_y = predict_y[0]
            loss_ = np.abs(predict_y - ty) / ty
            for i in range(len(loss_)):
                loss[i] = loss[i] + loss_[i]
            # print(predict_y, ty)
            f.write("\n" + str(predict_y) + " " + str(ty))
        time2 = time.time()
        loss /= (len(test_data_x)-self.stride-future+1)
        print("Net Model Test:")
        f.write("\n" + "Net Model Test:")
        print("---------------------------------------------------")
        f.write("\n" + "---------------------------------------------------")
        print("气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        f.write("\n" + "气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        print("H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        f.write("\n" + "H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        print("CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        f.write("\n" + "CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        print("CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        f.write("\n" + "CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        print("平均耗时：", (time2 - time1) / (len(test_data_x)-self.stride-future+1))
        f.write("\n" + "平均耗时：" + str((time2 - time1) / (len(test_data_x)-self.stride-future+1)))
        return [str(float(loss[0] * 100))[:8] + "%",str(float(loss[1] * 100))[:8] + "%",str(float(loss[2] * 100))[:8] + "%",str(float(loss[3] * 100))[:8] + "%",str((time2 - time1) / (len(test_data_x)-self.stride-future+1))]

    def online_train_LSTM(self,train_path, model_path, stride, EPOCH):
        print("-------------------------------------------")
        print("Online Train LSTM Model...")
        self.stride = stride
        self.EPOCH = EPOCH
        # Get train data and test data
        train_data_x, train_data_y = self.getOnlineData(data_files_dic=train_path,
                                                        normalized=True)

        # Print the size of train data and test data
        print("Train data shape:")
        print(train_data_x.shape)
        print(train_data_y.shape)

        train_data_x = torch.Tensor(train_data_x)
        train_data_y = torch.Tensor(train_data_y)

        trainset = DataSet(train_data_x, train_data_y)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=False)

        # Load the Net LSTM_model
        # net = torch.load('LSTM_model.pth')

        # Bulid the LSTM_model
        net = LSTM(self.input_size*self.stride, self.output_size)
        # Optimize all net parameters
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        # The loss function uses mean square error (MSE) loss function
        loss_func = nn.MSELoss()

        for step in range(self.EPOCH):
            total_loss = 0
            for tx, ty in trainloader:
                output = net.forward(torch.unsqueeze(tx, dim=0))
                # print(output.detach().numpy()[0][0],ty.numpy()[0])
                loss = loss_func(torch.squeeze(output), ty)
                # clear gradients for this training step
                optimizer.zero_grad()
                # back propagation, compute gradients
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            # print(step, float(total_loss))
        torch.save(net, model_path+"/LSTM_model_"+str(self.stride)+".pth")
        print("Save Online LSTM Model")

    def online_train_Net(self, train_path, model_path, stride, EPOCH):
        print("-------------------------------------------")
        print("Online Train Net Model...")
        self.stride = stride
        self.EPOCH = EPOCH
        # Get train data and test data
        train_data_x, train_data_y = self.getOnlineData(data_files_dic=train_path,
                                                        normalized=True)

        # Print the size of train data and test data
        print("Train data shape:")
        print(train_data_x.shape)
        print(train_data_y.shape)

        train_data_x = torch.Tensor(train_data_x)
        train_data_y = torch.Tensor(train_data_y)

        trainset = DataSet(train_data_x, train_data_y)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=False)

        # Bulid the Net_model
        net = Net(self.input_size * self.stride, self.output_size)
        # Optimize all net parameters
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        # The loss function uses mean square error (MSE) loss function
        loss_func = nn.MSELoss()

        for step in range(self.EPOCH):
            total_loss = 0
            for tx, ty in trainloader:
                output = net.forward(torch.unsqueeze(tx, dim=0))
                # print(output.detach().numpy()[0][0],ty.numpy()[0])
                loss = loss_func(torch.squeeze(output), ty)
                # clear gradients for this training step
                optimizer.zero_grad()
                # back propagation, compute gradients
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            # print(step, float(total_loss))
        torch.save(net, model_path + "/Net_model_" + str(self.stride) + ".pth")
        print("Save Online Net Model")

    def online_test_Net(self,test_path, model_path, stride):
        print("-------------------------------------------")
        print("Test Net Model...")

        self.stride = stride

        # Get train data and test data
        test_data_x, test_data_y = self.getOnlineData(data_files_dic=test_path, normalized=True)

        # Print the size of train data and test data
        print("Test data shape:")
        print(test_data_x.shape)
        print(test_data_y.shape)

        test_data_x = torch.Tensor(test_data_x)
        test_data_y = torch.Tensor(test_data_y)

        testset = DataSet(test_data_x, test_data_y)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        # Load the Net_model
        net = torch.load(model_path+"/Net_model_"+str(self.stride)+".pth")

        loss = np.array([0, 0, 0, 0], float)
        time1 = time.time()
        for tx, ty in testloader:
            output = net(torch.unsqueeze(tx, dim=0))
            output = torch.reshape(output, [output.shape[1], output.shape[2]])
            loss_ = torch.abs(output - ty) / ty
            loss_ = loss_.detach().numpy()[0]
            for i in range(len(loss_)):
                loss[i] = loss[i] + loss_[i]
            # print(output.detach().numpy(), ty.numpy())

        time2 = time.time()
        loss /= len(testloader)
        print("Net Model Test:")
        print("---------------------------------------------------")
        print("气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        print("H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        print("CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        print("CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        print("平均耗时：", (time2 - time1) / len(testloader))
        return [float(loss[0] * 100), float(loss[1] * 100), float(loss[2] * 100), float(loss[3] * 100), (time2 - time1) / len(testloader)]

    def online_test_LSTM(self,test_path, model_path, stride):
        print("-------------------------------------------")
        print("Test LSTM Model...")

        self.stride = stride

        # Get train data and test data
        test_data_x, test_data_y = self.getOnlineData(data_files_dic=test_path, normalized=True)

        # Print the size of train data and test data
        print("Test data shape:")
        print(test_data_x.shape)
        print(test_data_y.shape)

        test_data_x = torch.Tensor(test_data_x)
        test_data_y = torch.Tensor(test_data_y)

        testset = DataSet(test_data_x, test_data_y)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        # Load the Net_model
        net = torch.load(model_path+"/LSTM_model_"+str(self.stride)+".pth")

        loss = np.array([0, 0, 0, 0], float)
        time1 = time.time()
        for tx, ty in testloader:
            output = net(torch.unsqueeze(tx, dim=0))
            output = torch.reshape(output, [output.shape[1], output.shape[2]])
            loss_ = torch.abs(output - ty) / ty
            loss_ = loss_.detach().numpy()[0]
            for i in range(len(loss_)):
                loss[i] = loss[i] + loss_[i]
            # print(output.detach().numpy(), ty.numpy())

        time2 = time.time()
        loss /= len(testloader)
        print("LSTM Model Test:")
        print("---------------------------------------------------")
        print("气化温度误差率: " + str(float(loss[0] * 100))[:8] + "%")
        print("H2误差率: " + str(float(loss[1] * 100))[:8] + "%")
        print("CH4误差率: " + str(float(loss[2] * 100))[:8] + "%")
        print("CO误差率: " + str(float(loss[3] * 100))[:8] + "%")
        print("平均耗时：", (time2 - time1) / len(testloader))
        return [float(loss[0] * 100), float(loss[1] * 100), float(loss[2] * 100), float(loss[3] * 100), (time2 - time1) / len(testloader)]

if __name__ == '__main__':
    # 训练 & 测试
    # for i in [5,10,15,20,25,30,35,40,45,50]:
    #     model = Model(stride=i,EPOCH=1000)
    #     model.train_Net(normalized=True)
    #     model.test_Net(normalized=True)
    #
    #     model.train_LSTM(normalized=True)
    #     model.test_LSTM(normalized=True)

    # 测试未来一段时间
    A = []
    B = []
    for j in [10, 20, 30, 40, 50, 60]:
        for i in [5,10,15,20,25,30,35,40,45,50]:
            model = Model(stride=i, EPOCH=1000)
            # a = model.test_future_LSTM(normalized=True,future=j)
            b = model.test_future_Net(normalized=True,future=j)
