import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_output_size(input, kernel, stride):
    return (input - kernel) // stride + 1

class CNNNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels = params['data']['input_dim'][0]
        in_length = params['data']['input_dim'][1]
        width = params['data']['input_dim'][1]
        out_dim = params['data']['output_dim']

        if params is not None and 'fc_neurons' in params:
            fc_neurons = params['fc_neurons']
        else:
            fc_neurons = 90

        self.layers = nn.ModuleList()

        out_channels = params['channel_multiplier']

        if 'batch_norm' in params and params['batch_norm']:
            self.use_bias = True
            self.batch_norm = True
        else:
            self.use_bias = False
            self.batch_norm = False

        for _ in range(params['conv_layers']):
            if self.batch_norm:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=self.use_bias),
                    nn.BatchNorm1d(out_channels)))
            else:
                self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=self.use_bias))
            in_channels = out_channels
            out_channels *= 2
            width = width // 2

        if params is not None and 'dropout' in params:
            self.dropout = nn.Dropout(params['dropout'])
        self.flatten_size = out_channels//2 * width
        self.fc1 = nn.Linear(self.flatten_size, fc_neurons)
        self.fc2 = nn.Linear(fc_neurons, params['data']['output_dim'])

    def forward(self, x):
        for layer in self.layers:
            x = nn.ZeroPad1d((1,0))(x)
            x = F.relu(layer(x))

        x = x.view(-1, self.flatten_size)  # Flatten for the linear layers
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x