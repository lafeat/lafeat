import os
import argparse

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='attacks/cifar10_X.npy')
    parser.add_argument('--save-dir', type=str, default='attacks')
    parser.add_argument('--name', type=str, default='lafeat.pt')
    parser.add_argument('--epsilon', type=str, default='0.031')
    return parser.parse_args()


def permute(x):
    return x.permute(0, 2, 3, 1).contiguous()


def epsilon(s):
    # Different platforms and PyTorch/Numpy versions
    # could potentially give different
    # floating-point rounding errors,
    # this following hack tightens the epsilon
    # to ensure boundary within [-epsilon, epsilon].
    e = np.float32(s)
    return e - 4 * e * np.finfo(e).eps


def convert(args):
    x = np.float32(np.load(args.data))
    e = epsilon(args.epsilon)
    load_path = os.path.join(args.save_dir, args.name)
    xadv = torch.load(load_path)['adv_complete']
    xadv = permute(xadv)
    # d = torch.clamp(xadv - x, -e, e)
    # xadv = torch.clamp(d + x, 0.0, 1.0)
    xadv = xadv.numpy()
    # xadv = np.clip(xadv, x - e, x + e)
    d = np.clip(xadv - x, -e, e)
    xadv = np.clip(d + x, 0.0, 1.0)
    save_path = os.path.join(args.save_dir, 'cifar10_X_adv.npy')
    np.save(save_path, xadv)
    verbose(args.epsilon, xadv, x)


def verbose(eps, xadv, x):
    d = xadv - x
    oe = np.float64(eps)
    print(
        f'{d.min()}, {d.max()}, '
        f'True: {d.min() >= -oe}, True: {d.max() <= oe}.')


if __name__ == '__main__':
    convert(parse_args())
