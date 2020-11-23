import os
import sys
import argparse

import torch
from torch.utils import data
from torchvision import transforms, datasets

from utils import Logger
from frozen import Frozen
from wideresnet import WideResNet
from lafeat.eval import LafeatEval
from train import test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../datasets/cifar10')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--logits-model', type=str, default='models/trades.pt')
    parser.add_argument('--frozen-model', type=str, default='models/model_cifar_wrn.pt')
    parser.add_argument('--num-examples', type=int, default=10000)
    parser.add_argument('--save-dir', type=str, default='attacks')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--epsilon', type=float, default=0.031)
    parser.add_argument('--num-iterations', type=int, default=1)
    parser.add_argument('--multi-targeted', action='store_true')
    parser.add_argument('--beta-increment', type=float, default=0.1)
    return parser.parse_args()


def save(save_name, args, images, robust):
    os.makedirs(args.save_dir, exist_ok=True)
    name = f'lafeat.{int(robust.sum())}.{save_name.replace("lafeat.", "")}.pt'
    path = os.path.join(args.save_dir, name)
    result = {'adversarial_images': images, 'robust': robust}
    result.update(vars(args))
    torch.save(result, path)
    if args.verbose:
        print(f'Saved adversarial images at {path}.')


def main(args):
    save_name = f'lafeat.iter_{args.num_iterations}.eps_{args.epsilon:.5f}'
    if args.multi_targeted:
        save_name += '.mt'
    sys.stdout = Logger(os.path.join(args.log_dir, f'{save_name}.log'))

    device = torch.device('cuda')
    model = Frozen(WideResNet)
    state = torch.load(args.logits_model, map_location=device)
    model.load_state_dict(state['state_dict'], strict=False)
    model.load_frozen(args.frozen_model)
    model = model.cuda()
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(
        root=args.data_dir, train=False, transform=transform, download=True)
    test_loader = data.DataLoader(
        item, batch_size=1000, shuffle=False, num_workers=0)

    if args.verbose:
        accs, _ = test(model, test_loader, True)
        accs = ', '.join(f'{a:.2%}' for a in accs)
        print(f'Accuracies: {accs}')

    xtest, ytest = [torch.cat(xy, 0) for xy in zip(*test_loader)]
    if args.resume_from:
        path = os.path.join(args.save_dir, f'{args.resume_from}.pt')
        xtest = torch.load(path)['adversarial_images']
        # import numpy as np
        # from convert import convert
        # x = np.float32(np.load('attacks/cifar10_X.npy'))
        # xtest = torch.tensor(convert(x, xtest, np.float32('0.031')))
        # xtest = xtest.permute(0, 3, 1, 2).contiguous()
        if args.verbose:
            print(f'Resuming attack from {path}...')
    n = args.num_examples
    adversary = LafeatEval(
        model, xtest[:n], ytest[:n],
        n_iter=args.num_iterations, norm=args.norm, eps=args.epsilon,
        betas=(0, 1.0, args.beta_increment), target=args.multi_targeted,
        batch_size=args.batch_size, device=device, verbose=args.verbose)

    with torch.no_grad():
        images = adversary.eval()
        robust = adversary.robust
        save(save_name, args, images, robust)


if __name__ == '__main__':
    main(parse_args())
