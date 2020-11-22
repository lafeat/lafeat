# Essential Imports
import os
import sys
import argparse
import datetime
import time
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils import AverageMeter, Logger
from frozen import Frozen
from wideresnet import WideResNet


def parse_args():
    parser = argparse.ArgumentParser("Softmax Training for CIFAR-10 Dataset")
    parser.add_argument('--data-dir', type=str, default='../datasets/cifar10')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--load-name', type=str, default='models/model_cifar_wrn.pt')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--save-name', type=str, default='trades')
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
    parser.add_argument(
        '--weight-decay', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.stdout = Logger(os.path.join(args.log_dir, f'{args.save_name}.log'))
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
        root=args.data_dir,
        train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch, pin_memory=True,
        shuffle=True, num_workers=args.workers)

    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch, pin_memory=True,
        shuffle=False, num_workers=1)

    model = Frozen(WideResNet, widen_factor=10, num_classes=num_classes)
    model.load_frozen(args.load_name, device)
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
            accs, total = test(model, testloader, use_gpu)
            desc = ', '.join(
                f'Acc256_{i + 15}: {a:.2%}' for i, a in enumerate(accs[:-1]))
            print(f'{desc}, Acc_outputs: {accs[-1]:.2%}, Total: {total}')
    # save model
    checkpoint = {
        'epoch': args.max_epoch,
        'state_dict': model.state_dict(),
        'optimizer_model': optimizer.state_dict(),
    }
    path = os.path.join(args.save_dir, f'{args.save_name}.pt')
    torch.save(checkpoint, path)
    elapsed = round(time.time() - start_time)
    elapsed = datetime.timedelta(seconds=elapsed)
    print(f'Finished. Total elapsed time (h:m:s): {elapsed}')


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
    total = 0
    corrects = np.zeros(5)
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data = data.cuda()
            outputs = model(data)[-5:]
            predictions = np.array(
                [o.max(1)[1].cpu().numpy() for o in outputs])
            labels = labels.reshape(1, -1).detach().numpy()
            corrects += (predictions == labels).sum(1)
            total += labels.size
    accs = corrects / total
    return accs, total


if __name__ == '__main__':
    main()
