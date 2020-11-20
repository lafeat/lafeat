# Essential Imports
import os
import sys
import argparse
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils import AverageMeter, Logger
from frozen_wideresnet import FrozenWideResNet


def parse_args():
    parser = argparse.ArgumentParser("Softmax Training for CIFAR-10 Dataset")
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help="number of data loading workers (default: 4)")
    parser.add_argument(
        '--train-batch', default=128, type=int, metavar='N',
        help='train batch size')
    parser.add_argument(
        '--test-batch', default=100, type=int, metavar='N',
        help='test batch size')
    parser.add_argument(
        '--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--load-name', type=str, default='model_cifar_wrn.pt')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--save-name', type=str, default='trades')
    parser.add_argument(
        '--weight-decay', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.stdout = Logger(os.path.join(args.log_dir, '{args.load_name}.log'))
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    if use_gpu:
        print('Using GPU: {}'.format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print('Using CPU')

    os.makedirs(f'./{args.save_dir}', exist_ok=True)
    # Data Loading
    num_classes = 10
    print('==> Preparing dataset ')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root='../datasets/cifar10',
        train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch, pin_memory=True,
        shuffle=True, num_workers=args.workers)

    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='../datasets/cifar10', train=False,
        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch, pin_memory=True,
        shuffle=False, num_workers=1)

    model = FrozenWideResNet(
        filename=args.load_name, widen_factor=10,
        num_classes=num_classes, device=device)
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    model_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch, eta_min=1e-5)

    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print('LR: %f' % (model_lr.get_lr()[-1]))
        train(
            trainloader, model, criterion,
            optimizer, use_gpu, model_lr, args.print_freq)
        dotest = (
            args.eval_freq > 0 and
            epoch % args.eval_freq == 0 or
            (epoch + 1) == args.max_epoch)
        if dotest:
            print("==> Test")  # Tests after every eval_freq epochs
            accs, total = test(model, testloader, use_gpu)
            accs = '\t'.join(
                f'Acc256_{i + 15}: {a:.2%}' for i, a in enumerate(accs))
            print(f'{accs}\tTotal: {total}')
            if epoch + 1 == args.max_epoch:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_model': optimizer.state_dict(),
                }
                path = os.path.join(
                    args.save_dir, f'{args.save_name}_{epoch}.pth.tar')
                torch.save(checkpoint, path)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Finished. Total elapsed time (h:m:s): {}'.format(elapsed))


def train(trainloader, model, criterion, optimizer, use_gpu, model_lr, print_freq):
    model.train()
    meters = [AverageMeter() for _ in range(5)]
    all_meter = AverageMeter()

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        all_outputs = model(data)[-5:]

        loss_xent = [criterion(o, labels) for o in all_outputs]
        all_loss_xent = sum(loss_xent[:-1])

        optimizer.zero_grad()
        all_loss_xent.backward()
        optimizer.step()

        for l, m in zip(loss_xent, meters):
            m.update(l.item(), labels.size(0))
        all_meter.update(all_loss_xent.item(), labels.size(0))

        if (batch_idx + 1) % print_freq == 0:
            avgs = [
                f'Loss256_{i + 15}: {l.avg:.3f}' for i, l in enumerate(meters[:-1])]
            avgs = ', '.join(avgs)
            avgs += f', Loss_outputs {all_meter.avg:.3f}'
            print(f'Batch {batch_idx + 1}/{len(trainloader)}: {avgs}')
    model_lr.step()


def test(model, testloader, use_gpu):
    model.eval()
    total = 0, 0
    corrects = np.zeros(5)
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data = data.cuda()
            outputs = model(data)[-5:]
            predictions = np.array(
                [o.max(1)[1].detach().numpy() for o in outputs])
            labels = labels.reshape(1, -1).detach().numpy()
            corrects += (predictions == labels).sum(1)
            total += labels.size(0)
    accs = corrects / total
    return accs, total


if __name__ == '__main__':
    main()
