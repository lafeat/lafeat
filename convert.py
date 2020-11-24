import os
import argparse

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='attacks')
    parser.add_argument('--data', type=str, default='cifar10_X.npy')
    parser.add_argument('--name', type=str, default='lafeat.pt')
    parser.add_argument('--save-name', type=str, default='cifar10_X_adv.npy')
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


def convert(x, xadv, eps):
    e = epsilon(eps)
    xadv = permute(xadv)
    # d = torch.clamp(xadv - x, -e, e)
    # xadv = torch.clamp(d + x, 0.0, 1.0)
    xadv = xadv.numpy()
    # xadv = np.clip(xadv, x - e, x + e)
    d = np.clip(xadv - x, -e, e)
    return np.clip(d + x, 0.0, 1.0)


def verbose(eps, xadv, x):
    d = xadv - x
    oe = np.float64(eps)
    print(f'Min: {d.min()}. max: {d.max()}; ')
    print(f'Is min bound ok? {d.min() >= -oe}. ')
    print(f'Is max bound ok? {d.max() <= oe}.')


def main(args):
    x_path = os.path.join(args.save_dir, args.data)
    x = np.float32(np.load(x_path))
    xadv_path = os.path.join(args.save_dir, args.name)
    xadv = torch.load(xadv_path)['adversarial_images']
    xadv = convert(x, xadv, args.epsilon)
    save_path = os.path.join(args.save_dir, args.save_name)
    np.save(save_path, xadv)
    print(f'Saved converted images at {save_path}.')
    verbose(args.epsilon, xadv, x)


if __name__ == '__main__':
    main(parse_args())
