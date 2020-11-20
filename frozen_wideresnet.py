import torch.nn as nn
import torch

from wideresnet import WideResNet


class FrozenWideResNet(nn.Module):
    def __init__(
            self, filename='model_cifar_wrn.pt',
            widen_factor=10, num_classes=10, device=None):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        model = WideResNet(num_classes=num_classes)
        pretrained_checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(pretrained_checkpoint)
        model = nn.DataParallel(model)

        self.frozen_model = model
        for p in model.parameters():
            p.requires_grad = False

        self.avgpool = nn.AvgPool2d(8)

        self.fcf256_15_1 = nn.Linear(64 * widen_factor, 64 * widen_factor)
        self.fcf256_16_1 = nn.Linear(64 * widen_factor, 64 * widen_factor)
        self.fcf256_17_1 = nn.Linear(64 * widen_factor, 64 * widen_factor)
        self.fcf256_18_1 = nn.Linear(64 * widen_factor, 64 * widen_factor)

        self.fcf256_15 = nn.Linear(64 * widen_factor, num_classes)
        self.fcf256_16 = nn.Linear(64 * widen_factor, num_classes)
        self.fcf256_17 = nn.Linear(64 * widen_factor, num_classes)
        self.fcf256_18 = nn.Linear(64 * widen_factor, num_classes)

    def forward(self, x):
        self.frozen_model.eval()  # frozen model in eval mode
        f256_15, f256_16, f256_17, f256_18, f640, output_original = \
            self.frozen_model(x)[-6:]

        f256_15 = self.avgpool(f256_15)
        f256_16 = self.avgpool(f256_16)
        f256_17 = self.avgpool(f256_17)
        f256_18 = self.avgpool(f256_18)

        f256_15 = f256_15.view(f256_15.size(0), -1)
        f256_16 = f256_16.view(f256_16.size(0), -1)
        f256_17 = f256_17.view(f256_17.size(0), -1)
        f256_18 = f256_18.view(f256_18.size(0), -1)

        f256_15_640 = self.fcf256_15_1(f256_15)
        f256_16_640 = self.fcf256_16_1(f256_16)
        f256_17_640 = self.fcf256_17_1(f256_17)
        f256_18_640 = self.fcf256_18_1(f256_18)

        output_256_15 = self.fcf256_15(f256_15_640)
        output_256_16 = self.fcf256_16(f256_16_640)
        output_256_17 = self.fcf256_17(f256_17_640)
        output_256_18 = self.fcf256_18(f256_18_640)
        all_outputs = [
            f256_15_640, f256_16_640, f256_17_640, f256_18_640, f640,
            output_256_15, output_256_16, output_256_17, output_256_18,
            output_original]
        return all_outputs
