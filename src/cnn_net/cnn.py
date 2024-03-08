import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_output_size(input, kernel, stride):
    return (input - kernel) // stride + 1

class CNNNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_ch = config['data']['input_dim']
        out_dim = config['data']['output_dim']
        in_length = config['data']['input_length']

        self.conv_layers = []
        self.fc_layers = []
        out_channels = 0

        for layer in config['model']['layers']:
            if layer['type'] == 'Conv1D':
                self.conv_layers.append(nn.Conv1d(in_ch, layer['filters'], layer['kernel_size'], layer['stride']))
                in_ch = layer['filters']
                in_length = conv_output_size(in_length, layer['kernel_size'], layer['stride'])
                out_channels = layer['filters']
            elif layer['type'] == 'Flatten':
                self.fc_layers.append(nn.Flatten())
                out_neurons = in_length * out_channels
            elif layer['type'] == 'Dense':
                self.fc_layers.append(nn.Linear(out_neurons, layer['units']))
                out_neurons = layer['units']

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.fc_layers = nn.ModuleList(self.fc_layers)

        self.out = nn.Linear(out_neurons, out_dim)

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        for layer in self.fc_layers:
            x = F.relu(layer(x))

        x = self.out(x)
        return x
