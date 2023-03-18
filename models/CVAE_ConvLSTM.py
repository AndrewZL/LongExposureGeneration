import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from AutoEncoder_ConvLSTM import ConvLSTMCell
from torch.distributions import Normal


class MotionEncoder(nn.Module):
    def __init__(self, in_c=2, out_c=8):
        super(MotionEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        )
        self.fc = nn.Linear(256, out_c)  # latent
        self.fc_variance = nn.Linear(256, out_c)

    def forward(self, x):
        x_c = self.conv_layers(x)
        x_flatten = x_c.view(x.size(0), -1)
        mu = self.fc(x_flatten)
        var = self.fc_variance(x_flatten)
        return mu, var


class CVAE(nn.Module):
    def __init__(self, nf, in_c):
        super(AEConvLSTM, self).__init__()
        self.nf = nf
        self.e1 = ConvLSTMCell(in_c, nf, 3)
        self.e2 = ConvLSTMCell(nf, nf, 3)
        self.d1 = ConvLSTMCell(nf, nf, 3)
        self.d2 = ConvLSTMCell(nf, nf, 3)
        self.d3 = nn.Conv3d(nf, 3, 3, 1)

    def forward(self, x, future_seq=10, hidden_state=None):
        b, seq_len, c, h, w = x.size()

        out1, hidden1 = self.e1.init_hidden(b, (h, w))
        out2, hidden2 = self.e2.init_hidden(b, (h, w))
        out3, hidden3 = self.d1.init_hidden(b, (h, w))
        out4, hidden4 = self.d2.init_hidden(b, (h, w))
        outputs = []
        for t in range(seq_len):
            out1, hidden1 = self.e1(x[:, t, :, :], [out1, hidden1])
            out2, hidden2 = self.e2(out2, [out2, hidden2])

        mu = self.mu(hidden2[1])
        log_var = self.log_var(hidden2[1])
        gaussian_noise = Normal(b, self.nf).cuda()
        print(gaussian_noise.shape, log_var.shape, mu.shape)
        z = gaussian_noise * torch.exp(0.5 * log_var) + mu

        for t in range(future_step):
            out3, hidden3 = self.decoder_1_convlstm(z, [out3, hidden3])
            out4, hidden4 = self.decoder_2_convlstm(out3, [out4, hidden4])
            out4 = out4
            outputs += [out4]
        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)
        return outputs

