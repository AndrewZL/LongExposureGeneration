import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from data.datasets import SkyDataset
from models.CAE_ConvLSTM import CAE
from models.common import MotionEncoder

from tqdm import tqdm


def train_net(me, cae, optim_me, optim_cae, flow_model, flow_tf, train_dataloader, num_epochs, device):
    criterion = nn.MSELoss()
    train_loss = []

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for inputs, gts in train_dataloader:
            inputs = inputs.to(device)
            gts = gts.to(device)

            # find flow
            b1 = inputs[:, 0]
            b2 = inputs[:, 1]
            b1, b2 = flow_tf(b1, b2)
            flow = torch.stack(flow_model(b1, b2))
            flow = flow[-1, :]
            flow = F.interpolate(flow, size=(128, 128), mode='bilinear', align_corners=True)
            z = me(flow)

            # stack
            z_matched = z.view(z.size(0), 1, z.size(1), 1, 1).expand(inputs.size(0), inputs.size(1), z.size(1),
                                                                     inputs[0].size(2), inputs[0].size(3))
            inputs = torch.cat((inputs, z_matched), 2)

            # prediction
            preds = cae(inputs).permute(0, 2, 1, 3, 4)
            optim_me.zero_grad()
            optim_cae.zero_grad()

            loss = criterion(preds, gts)
            loss += criterion((preds[:, -1] - preds[:, 0]).detach(), (gts[:, -1] - gts[:, 0]).detach())

            loss.backward()
            optim_me.step()
            optim_cae.step()

            total_loss += loss.item()

        train_loss.append(float(total_loss) / len(train_dataloader))
        print("Epoch {}: Train loss: {}".format(epoch + 1, train_loss[epoch]))

        torch.save({
            'epoch': epoch,
            'me': me.state_dict(),
            'cae': cae.state_dict(),
            'optim_me': optim_me.state_dict(),
            'optim_cae': optim_cae.state_dict()
        },
            f'/kaggle/working/checkpoint_cae_{epoch}.pt')
        np.savetxt(f'loss_{epoch}.txt', train_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../SkyDataset/train')
    parser.add_argument('--gt_dir', type=str, default='../SkyDataset/gt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_cae_302.pt')
    args = parser.parse_args()

    train_set = SkyDataset(args.input_dir, args.gt_dir)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    weights = Raft_Small_Weights.DEFAULT
    flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
    flow_model = flow_model.eval()
    flow_tf = weights.transforms()

    lr = args.lr
    num_epochs = args.num_epochs
    img_w = args.img_w
    img_h = args.img_h

    me = MotionEncoder()
    me = me.to(device)
    cae = CAE(nf=22, in_chan=11)
    cae = cae.to(device)

    optim_me = Adam(me.parameters(), lr=lr)
    optim_cae = Adam(cae.parameters(), lr=lr)

    checkpoint = torch.load(args.checkpoint)
    me.load_state_dict(checkpoint['me'])
    cae.load_state_dict(checkpoint['cvae'])  # checkpoints contain a mistake in naming here
    optim_me.load_state_dict(checkpoint['optim_me'])
    optim_cae.load_state_dict(checkpoint['optim_cvae'])

    train_net(me, cae, optim_me, optim_cae, flow_model, flow_tf, train_dataloader, num_epochs, device)