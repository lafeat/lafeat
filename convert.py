import os
import argparse

import torch
from torch.utils import data
from torchvision import transforms, datasets
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='attacks/cifar10_X.npy')
    parser.add_argument('--save-dir', type=str, default='attacks')
    parser.add_argument('--name', type=str, default='lafeat.pt')
    return parser.parse_args()


def permute(x):
    return x.permute(0, 2, 3, 1).contiguous()


def convert(args):
    x = np.load(args.data)
    epsilon = 0.031
    # Different platforms and PyTorch/Numpy versions
    # could potentially give different
    # floating-point rounding errors,
    # this following line is a hack
    # to ensure boundary within epsilon.
    # epsilon -= np.finfo(np.float32(1.0)).eps
    load_path = os.path.join(args.save_dir, args.name)
    xadv = torch.load(load_path)['adv_complete'].float()
    xadv = permute(xadv)
    d = torch.clamp(xadv - x, -epsilon, epsilon)
    xadv = torch.clamp(d + x, 0.0, 1.0)
    save_path = os.path.join(args.save_dir, 'cifar10_X_adv.npy')
    np.save(save_path, xadv.numpy())


if __name__ == '__main__':
    convert(parse_args())
