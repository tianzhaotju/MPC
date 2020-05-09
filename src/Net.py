from torch import nn

class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Net,self).__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid()
        )
        self.net = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(16,output_size)
        )

    def forward(self, x):
        net_in = self.input(x)
        net_out = self.net(net_in)
        out = self.out(net_out)
        return out
