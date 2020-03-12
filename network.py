from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from circles_experiment.tcn import TemporalConvNet


class LSTM(nn.Module):
    """
        输入：
            [batch, seq_len, input_size]

        输出:
            [batch, seq_len, input_size]
    """

    def __init__(self, input_size, hidden_size, hidden_size2, device='cpu'):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=hidden_size2)
        self.linear2 = nn.Linear(in_features=hidden_size2, out_features=2)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, lengths):
        # 打包可以加速可变长序列的计算 (不会影响计算结果)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # self.lstm.flatten_parameters()
        hidden_out, (h_n, c_n) = self.lstm(x)  # hidden_out: [batch_size, seq_len, hidden_size]
        # 还原打包
        hidden_out, _ = pad_packed_sequence(hidden_out, batch_first=True)
        _Y = self.linear1(hidden_out)  # _Y: [batch_size, seq_len, 2]
        _Y = F.relu(_Y)
        _Y = self.linear2(_Y)
        return _Y


class TCN(nn.Module):
    """
        输入：
            [batch, seq_len, input_size]

        输出:
            [batch, seq_len, input_size]
    """

    def __init__(self, input_size, hidden_size, hidden_size2, device='cpu'):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=[hidden_size] * 6)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=hidden_size2)
        self.linear2 = nn.Linear(in_features=hidden_size2, out_features=2)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        hidden_out = self.tcn(x)  # hidden_out: [batch_size, seq_len, hidden_size]
        hidden_out = hidden_out.transpose(1, 2)
        _Y = self.linear1(hidden_out)  # _Y: [batch_size, seq_len, 2]
        _Y = F.relu(_Y)
        _Y = self.linear2(_Y)
        return _Y
