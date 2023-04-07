import os
import re

import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class SkyDataset(Dataset):
    """
    Dataset class for the "Sky Timelapse Dataset" that returns an input sequence (initial 5 frames)
    and ground truth sequence (10 frames that follow the input sequence)
    """
    def __init__(self, train_dir, gt_dir):
        self.train_root = train_dir
        self.gt_root = gt_dir
        self.train_dirs = sorted(os.listdir(train_dir))
        self.gt_dirs = sorted(os.listdir(gt_dir))

    def __len__(self):
        return len(self.train_dirs)

    def __getitem__(self, idx):
        train_seq = torch.stack([read_image(os.path.join(self.train_root, self.train_dirs[idx], x)) / 255.0 for x in
                                 sorted(os.listdir(os.path.join(self.train_root, self.train_dirs[idx])))])
        gt_seq = torch.stack([read_image(os.path.join(self.gt_root, self.gt_dirs[idx], x)) / 255.0 for x in
                              sorted(os.listdir(os.path.join(self.gt_root, self.gt_dirs[idx])))])
        return train_seq, gt_seq


class SkyDataset_old(Dataset):
    """
    Dataset class for the "Sky Timelapse Dataset" that returns an input sequence (initial 5 frames)
    and ground truth long exposure
    """

    def __init__(self, train_dir, gt_dir):
        self.train_dir = train_dir
        self.gt_dir = gt_dir

        self.gt_imgs = os.listdir(gt_dir)  # image filename
        self.gt_paths = [os.path.join(self.gt_dir, self.gt_imgs[x]) for x in range(len(self.gt_imgs))]  # total path

        self.gt = []
        self.train = []

        pattern = r'^([A-Za-z0-9_-]+)\_\d+\.jpg$'
        for idx, gt_img_fname in enumerate(self.gt_imgs):
            match = re.search(pattern, gt_img_fname)
            if match:
                data_folder = match.group(1)
                data_imgs = sorted(os.listdir(os.path.join(self.train_dir, data_folder, gt_img_fname[:-4])))
                data_paths = [os.path.join(self.train_dir, data_folder, gt_img_fname[:-4], x) for x in data_imgs]
                if len(data_paths) > 5:
                    self.gt.append(self.gt_paths[idx])
                    self.train.append(data_paths)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        train_seq = torch.stack([read_image(x) / 255.0 for x in self.train[idx][:5]])
        gt_image = read_image(self.gt[idx]) / 255.0

        return train_seq, gt_image


class SkyDatasetSequence_old(Dataset):
    """
    Dataset class for the "Sky Timelapse Dataset" that returns the entire sequence
    and ground truth long exposure
    """

    def __init__(self, train_dir, gt_dir):
        self.train_dir = train_dir
        self.gt_dir = gt_dir

        self.gt_imgs = os.listdir(gt_dir)  # image filename
        self.gt_paths = [os.path.join(self.gt_dir, self.gt_imgs[x]) for x in range(len(self.gt_imgs))]  # total path

        self.gt = []
        self.train = []

        pattern = r'^([A-Za-z0-9_-]+)\_\d+\.jpg$'
        for idx, gt_img_fname in enumerate(self.gt_imgs):
            match = re.search(pattern, gt_img_fname)
            if match:
                data_folder = match.group(1)
                data_imgs = sorted(os.listdir(os.path.join(self.train_dir, data_folder, gt_img_fname[:-4])))
                data_paths = [os.path.join(self.train_dir, data_folder, gt_img_fname[:-4], x) for x in data_imgs]
                if len(data_paths) > 5:
                    self.gt.append(self.gt_paths[idx])
                    self.train.append(data_paths)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        gt_seq = torch.stack([read_image(x) / 255.0 for x in self.train[idx]])
        gt_image = read_image(self.gt[idx]) / 255.0

        return gt_seq, gt_image
