from torch import nn

class LSTM(nn.Module):
    def __init__(self,input_size,output_size):
        super(LSTM,self).__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=16,
            num_layers=2
        )
        self.out = nn.Sequential(
            nn.Linear(16,output_size)
        )

    def forward(self, x):
        lstm_in = self.input(x)
        lstm_out, (h_n,h_c) = self.lstm(lstm_in)
        out = self.out(lstm_out)
        return out

