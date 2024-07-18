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
        self.width = config['data']['input_dim'][1]
        out_ch = config['data']['output_dim']
        self.classes = out_ch
        hidden = config.get('fc_neurons', 64)

        self.depth = config.get('depth', 3)
        if self.depth == 1: # 10 layers
            l1, l2, l3, l4 = 1, 1, 2, 1
        elif self.depth == 2: # 20 layers
            l1, l2, l3, l4 = 2, 3, 3, 2
        elif self.depth == 3: # 30 layers
            l1, l2, l3, l4 = 3, 4, 5, 3
        elif self.depth == 4:
            l1, l2, l3, l4 = 3, 5, 6, 3 # 6+8+5=38
        else:
            raise ValueError('Invalid depth value')

        base = 8
        self.channel_multiplier = config.get('channel_multiplier', 1)
        base_mlt = base*self.channel_multiplier

        self.initial_conv = nn.Conv2d(in_ch, base_mlt, 7, 2, padding=3)

        self.res_blocks = nn.ModuleList([
            nn.ModuleList(self._make_stage(base_mlt, base_mlt, 1, l1)),
            nn.ModuleList(self._make_stage(base_mlt, 2*base_mlt, 2, l2)),
            nn.ModuleList(self._make_stage(2*base_mlt, 4*base_mlt, 2, l3)),
            nn.ModuleList(self._make_stage(4*base_mlt, 8*base_mlt, 2, l4)),
        ])

        self.final_layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=(int(self.width/32), int(self.width/32))),
            nn.Flatten(),
            nn.Linear(8*base_mlt, hidden),  # Adjust input features if needed
            nn.ReLU(),
            nn.Linear(hidden, out_ch)
        )

    def _make_stage(self, in_ch, out_ch, stride, num_blocks):
        return [ResidualBlock(in_ch, out_ch, stride)] + \
            [ResidualBlock(out_ch, out_ch) for _ in range(num_blocks - 1)]

    def forward(self, x):
        #x = F.pad(x, (0,1,0,1))
        x = F.relu(self.initial_conv(x))

        x = F.pad(x, (0,1,0,1))
        x = F.max_pool2d(x, 3, 2)

        for block in self.res_blocks:
            for res_block in block:
                x = res_block(x)

        x = self.final_layers(x)
        return x
