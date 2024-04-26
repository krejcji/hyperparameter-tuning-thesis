import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_output_size(input, kernel, stride):
    return (input - kernel) // stride + 1

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.stride = stride
        kernel_size = 3
        padding = 'same'

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.ModuleList()
        if stride != 1 or in_ch != out_ch:
            self.shortcut.append(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))
            self.shortcut.append(nn.BatchNorm2d(out_ch))
        else:
            self.shortcut.append(nn.Identity())

    def forward(self, x):
        identity = x

        for layer in self.shortcut:
            identity = layer(identity)
        if self.stride == 1:
            out = F.pad(x, (1,1,1,1))
        elif self.stride == 2:
            out = F.pad(x, (0,1,0,1))

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class CNN2DNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_ch = config['data']['input_dim'][0]
        out_ch = config['data']['output_dim']
        self.classes = out_ch

        self.initial_conv = nn.Conv2d(in_ch, 16, 3, 1, padding='same')

        self.res_blocks = nn.ModuleList([
            nn.ModuleList(self._make_stage(16, 16, 1, 3)),
            nn.ModuleList(self._make_stage(16, 32, 2, 3)),
            nn.ModuleList(self._make_stage(32, 32, 1, 3)),
            nn.ModuleList(self._make_stage(32, 64, 2, 3)),
            nn.ModuleList(self._make_stage(64, 64, 1, 3))
        ])

        self.final_layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=(8, 8)),
            nn.Flatten(),
            nn.Linear(64, 10),  # Adjust input features if needed
            nn.ReLU(),
            nn.Linear(10, out_ch)
        )

    def _make_stage(self, in_ch, out_ch, stride, num_blocks):
        return [ResidualBlock(in_ch, out_ch, stride)] + \
            [ResidualBlock(out_ch, out_ch) for _ in range(num_blocks - 1)]

    def forward(self, x):
        x = x / 255
        x = F.relu(self.initial_conv(x))

        for block in self.res_blocks:
            for res_block in block:
                x = res_block(x)

        x = self.final_layers(x)
        return x
