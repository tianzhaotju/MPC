class DataSet():
    def __init__(self,data_x,data_y):
        self.data, self.label = data_x.float(), data_y.float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
