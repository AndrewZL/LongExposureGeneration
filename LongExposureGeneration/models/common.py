import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    def __init__(self, in_c=2, out_c=8):
        super(MotionEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
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
            nn.Conv2d(256, 256, kernel_size=8, stride=8, padding=0),
            nn.LeakyReLU()
        )
        if in_c > 2:
            in_dim = 2560
        else:
            in_dim = 256
        self.fc = nn.Linear(in_dim, out_c)  # latent
        self.fc_variance = nn.Linear(in_dim, out_c)

    def forward(self, x):
        x_c = self.conv_layers(x)
        x_flatten = x_c.view(x.size(0), -1)
        out = self.fc(x_flatten)
        var = self.fc_variance(x_flatten)
        return out, var


class MotionDecoder(nn.Module):
    def __init__(self, in_c=1280, out_c=64):
        super(MotionDecoder, self).__init__()
        self.fc = nn.Linear(in_c, 2560)  # latent

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=8, stride=8, padding=0, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, out_c, kernel_size=5, stride=2, padding=2, output_padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 2, 5)
        x_c = self.conv_layers(x)
        # upsample to match the original input size
        x_c = F.interpolate(x_c, size=(360, 640), mode='bilinear', align_corners=True)
        return x_c


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width),
                torch.zeros(batch_size, self.hidden_dim, height, width))
