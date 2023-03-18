import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell based on https://holmdk.github.io/2020/04/02/video_prediction.html
    """

    def __init__(self, input_dim, nf, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.nf = nf
        padding = kernel_size[0] // 2
        self.conv = nn.Conv2d(self.input_dim + self.nf, 4 * self.nf, kernel_size, padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        x = torch.cat([input_tensor, h_cur], dim=1)
        x = self.conv(x)

        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.nf, dim=1)
        i = F.LeakyRelu(cc_i)
        f = F.LeakyRelu(cc_f)
        o = F.LeakyRelu(cc_o)
        g = F.LeakyRelu(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.LeakyRelu(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class AEConvLSTM(nn.Module):
    def __init__(self, nf, in_c):
        super(AEConvLSTM, self).__init__()
        self.nf = nf
        self.e1 = ConvLSTMCell(in_c, nf, 3)
        self.e2 = ConvLSTMCell(nf, nf, 3)
        self.d1 = ConvLSTMCell(nf, nf, 3)
        self.d2 = ConvLSTMCell(nf, nf, 3)
        self.d3 = nn.Conv3d(nf, 3, 3, 1)

    def forward(self, x, preds, hidden_state=None):
        b, seq_len, c, h, w = x.size()

        out1, hidden1 = self.e1.init_hidden(b, (h, w))
        out2, hidden2 = self.e2.init_hidden(b, (h, w))
        out3, hidden3 = self.d1.init_hidden(b, (h, w))
        out4, hidden4 = self.d2.init_hidden(b, (h, w))

        outputs = []
        for t in range(seq_len):
            out1, hidden1 = self.e1(x[:, t, :, :], [out1, hidden1])
            out2, hidden2 = self.e2(out2, [out2, hidden2])
        for t in range(preds):
            out3, hidden3 = self.decoder_1_convlstm(out2, [out3, hidden3])
            out4, hidden4 = self.decoder_2_convlstm(out3, [out4, hidden4])
            outputs += [out4]  
        outputs = torch.stack(outputs, 1)
        outputs = self.decoder_CNN(outputs.permute(0, 2, 1, 3, 4))
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs
