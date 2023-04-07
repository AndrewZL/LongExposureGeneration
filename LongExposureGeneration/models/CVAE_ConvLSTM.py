import torch
import torch.nn as nn
from LongExposureGeneration.models.common import MotionEncoder, MotionDecoder, ConvLSTMCell


class CVAE(nn.Module):
    def __init__(self, nf, in_chan):
        super(CVAE, self).__init__()
        self.nf = nf
        self.e1 = ConvLSTMCell(input_dim=in_chan, hidden_dim=nf, kernel_size=3)
        self.e2 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=3)
        self.e_VAE = MotionEncoder(in_c=10, out_c=1280)
        self.d_VAE = MotionDecoder(in_c=1280, out_c=10)
        self.d1 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=3)
        self.d2 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).cuda()
        return mu + noise * std

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.e1(input_tensor=x[:, t, :, :], cur_state=[h_t, c_t])
            h_t2, c_t2 = self.e2(input_tensor=h_t, cur_state=[h_t2, c_t2])
        mu, logvar = self.e_VAE(h_t2)
        z = self.reparameterize(mu, logvar)
        z = self.d_VAE(z)
        h_t2 = z
        for t in range(future_step):
            h_t3, c_t3 = self.d1(input_tensor=h_t2, cur_state=[h_t3, c_t3])
            h_t4, c_t4 = self.d2(input_tensor=h_t3, cur_state=[h_t4, c_t4])
            outputs += [h_t4]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)
        return outputs

    def forward(self, x, future_seq=10):
        b, seq_len, c, h, w = x.size()

        h_t, c_t = self.e1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.e2.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.d1.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.d2.init_hidden(batch_size=b, image_size=(h, w))

        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        return outputs
