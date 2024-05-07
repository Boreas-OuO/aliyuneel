import torch
import torch.nn as nn

class DFTCNN(nn.Module):
    def __init__(self, seq_length, in_channels, out_channels, kernel_size):
        super(DFTCNN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        # 在初始化时不指定线性层的输入大小
        self.linear_input_size = out_channels * seq_length * seq_length
        self.linear = nn.Linear(self.linear_input_size, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # 获取卷积层输出的特征图大小
        batch_size, _, _, _ = x.size()
        # # 动态计算线性层的输入大小
        # linear_input_size = channels * height * width
        #
        # # 更新线性层的输入大小
        # self.linear.in_features = linear_input_size

        x = x.reshape(batch_size, -1)
        x = self.linear(x)

        return x
class DFTALEX(nn.Module):
    def __init__(self, seq_length, in_channels, out_channels, kernel_size,p=0.5):
        super(DFTALEX, self).__init__()
        self.out_channels = out_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.out_channels * 2 * 3 * 3, 2 * 3 * 3),
            nn.ReLU(inplace=True),
            nn.Linear(2 * 3 * 3, 3 * 3),
            nn.ReLU(inplace=True),
            nn.Linear(3 * 3, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#-----------------------------------------RNN-----------------------------------------
class RNN(nn.Module):
    # 暂时跑不通，需要看一下输入输出的数据格式
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super().__init__()
        self.num_features = num_features 
        
        # Defining the number of layers and the nodes in each layer
        self.hidden_units = hidden_units
        self.num_layers = num_layers       
        self.output_size = output_size
        self.dropout = dropout_rate

        # RNN layers
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate           
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.output_size)

    def forward(self, x):
        # (batch_size, seq_length, feature)
        batch_size = x.shape[0]       
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()       
        _, hn = self.rnn(x, h0)
                
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out




# -----------------------------------------LSTM-----------------------------------------
class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super().__init__()
        self.num_features = num_features  #

        # Defining the number of layers and the nodes in each layer
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout_rate

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate
        )
        # Fully connected layer
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above
        return out



#-----------------------------------------GRU-----------------------------------------
class GRU(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.fc_out = nn.Linear(hidden_units, output_size)

        self.d_feat = num_features

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc_out(out[:, -1, :]).flatten()
        return out


#-----------------------------------------CNN_LSTM-----------------------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, device, in_channels, out_channels, sequence_length, kernel_size, batch_size, hidden_units, num_layers, output_size, dropout_rate):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_units
        self.num_layers = num_layers
        self.device = device
        # CNN模型
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)  # 16*12*12
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 16*1*1
        # LSTM模型
        input_size = out_channels*6*6
        self.input_size = input_size
        # self.out_channels = out_channels
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate)
        self.fc = nn.Linear(hidden_units, output_size)

    # 这里forward函数之前多放了一个参数_
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], 1, -1)
        batch_size = x.shape[0]
        # print("二维CNN输出数据形状:", x.shape)
        # 把CNN输出的feature压平，输入LSTM里面
        # x = x.reshape(1, x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)  # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)   # 初始化记忆状态c0
        # print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :]).flatten()  # 取最后一个时间步的输出作为预测结果
        return out


# -----------------------------------------ALSTM-----------------------------------------
class ALSTMModel(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super().__init__()
        self.hid_size = hidden_units
        self.input_size = num_features
        self.dropout = dropout_rate
        self.output_size = output_size
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):

        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())

        self.rnn = nn.LSTM(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )

        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=self.output_size, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return out.flatten()






