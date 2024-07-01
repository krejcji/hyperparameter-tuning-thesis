import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2DSimpleNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layers = nn.ModuleList()

        in_channels = params['data']['input_dim'][0]
        out_channels = params.get('channel_multiplier', 1)
        width = params['data']['input_dim'][1]

        batch_norm = params.get('batch_norm', False)
        fc_neurons = params.get('fc_neurons', 90)

        for _ in range(params['conv_layers']):
            if batch_norm:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=(1,1), bias=False),
                    nn.BatchNorm2d(out_channels)))
            else:
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=(1,1),bias=True))
            in_channels = out_channels
            out_channels *= 2
            width = width // 2

        if 'dropout' in params:
            self.dropout = nn.Dropout(params['dropout'])
        self.flatten_size = out_channels//2 * width * width
        self.fc1 = nn.Linear(self.flatten_size, fc_neurons)
        self.fc2 = nn.Linear(fc_neurons, params['data']['output_dim'])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        x = x.view(-1, self.flatten_size)  # Flatten for the linear layers
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x