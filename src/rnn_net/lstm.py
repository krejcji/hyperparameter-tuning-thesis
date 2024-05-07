import torch
from torch import nn

class LSTM(nn.Module):
  def __init__(self, params):
    super(LSTM, self).__init__()
    in_channels = params['data']['input_dim'][0]
    self.output_size = params['data']['output_dim']
    self.hidden_size = params['rnn_hidden']
    self.num_layers = params['rnn_layers']
    self.lstm = nn.LSTM(in_channels, self.hidden_size, self.num_layers, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, params['data']['output_dim'])
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #self.pool = nn.AdaptiveConcatPool1d(output_size=1)

  def forward(self, x):
    # Pass data through LSTM layers

    batch_size = x.shape[0]
    x = x.permute(0, 2, 1)
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

    h0 = h0.to(self.device)
    c0 = c0.to(self.device)

    _, (lstm_out, _) = self.lstm(x, (h0, c0))

    # Pass the output of the last time step to the fully connected layer
    out = self.fc(lstm_out[0])
    return out

    # Concatenate outputs from all layers
    pooled_output = self.pool(lstm_out)
    return pooled_output