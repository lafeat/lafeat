import torch.nn as nn
import torch
import torch.nn.functional as F


class Frozen(nn.Module):
    def __init__(self, model_cls, widen_factor=10, num_classes=10):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.avgpool = nn.AvgPool2d(8)
        # 8-->4
        self.conv15_8_4 = nn.Conv2d(64 * widen_factor, 64, kernel_size=2, stride=2, bias=False)
        self.conv16_8_4 = nn.Conv2d(64 * widen_factor, 64, kernel_size=2, stride=2, bias=False)
        self.conv17_8_4 = nn.Conv2d(64 * widen_factor, 64, kernel_size=2, stride=2, bias=False)
        self.conv18_8_4 = nn.Conv2d(64 * widen_factor, 64, kernel_size=2, stride=2, bias=False)
        # 4-->2
        self.conv15_4_2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)
        self.conv16_4_2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)
        self.conv17_4_2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)
        self.conv18_4_2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)

        self.fcf256_15_1 = nn.Linear(64 * 2 * 2, 256)
        self.fcf256_16_1 = nn.Linear(64 * 2 * 2, 256)
        self.fcf256_17_1 = nn.Linear(64 * 2 * 2, 256)
        self.fcf256_18_1 = nn.Linear(64 * 2 * 2, 256)

        self.fcf256_15 = nn.Linear(256, 10)
        self.fcf256_16 = nn.Linear(256, 10)
        self.fcf256_17 = nn.Linear(256, 10)
        self.fcf256_18 = nn.Linear(256, 10)

    def load_frozen(self, filename, device=None):
        state = torch.load(filename, map_location=device)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')}

    def forward(self, x):
        self.frozen_model.eval()  # frozen model in eval mode
        f256_15, f256_16, f256_17, f256_18, f640, output_original = \
            self.frozen_model(x)[-6:]

        # 8-->4
        fconv15_8_4 = F.relu(self.conv15_8_4(f256_15))
        fconv16_8_4 = F.relu(self.conv16_8_4(f256_16))
        fconv17_8_4 = F.relu(self.conv17_8_4(f256_17))
        fconv18_8_4 = F.relu(self.conv18_8_4(f256_18))
        # 4-->2
        fconv15_2 = F.relu(self.conv15_4_2(fconv15_8_4))
        fconv16_2 = F.relu(self.conv16_4_2(fconv16_8_4))
        fconv17_2 = F.relu(self.conv17_4_2(fconv17_8_4))
        fconv18_2 = F.relu(self.conv18_4_2(fconv18_8_4))

        f256_15 = fconv15_2.view(fconv15_2.size(0), -1)
        f256_16 = fconv16_2.view(fconv16_2.size(0), -1)
        f256_17 = fconv17_2.view(fconv17_2.size(0), -1)
        f256_18 = fconv18_2.view(fconv18_2.size(0), -1)

        f256_15_1 = F.relu(self.fcf256_15_1(f256_15))
        f256_16_1 = F.relu(self.fcf256_16_1(f256_16))
        f256_17_1 = F.relu(self.fcf256_17_1(f256_17))
        f256_18_1 = F.relu(self.fcf256_18_1(f256_18))

        output_256_15 = self.fcf256_15(f256_15_1)
        output_256_16 = self.fcf256_16(f256_16_1)
        output_256_17 = self.fcf256_17(f256_17_1)
        output_256_18 = self.fcf256_18(f256_18_1)
        all_outputs = [output_256_15, output_256_16, output_256_17, output_256_18, output_original]
        return all_outputs
