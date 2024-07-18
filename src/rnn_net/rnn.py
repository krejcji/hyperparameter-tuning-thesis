"""
Implementation of the recurrent neural network in PyTorch.
Supports LSTM and GRU cells.
"""
import torch
from torch import nn

class RNN(nn.Module):
  def __init__(self, params):
    super(RNN, self).__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_channels = params['data']['input_dim'][0]
    self.output_size = params['data']['output_dim']
    self.hidden_size = params['rnn_hidden']
    self.num_layers = params['rnn_layers']
    self.rnn_type = params['rnn_type']

    rnn_dropout = params.get('rnn_dropout', 0.0)
    bidirectional = params.get('bidirectional', 0)
    self.bidirectional = True if bidirectional == 1 else False
    if self.bidirectional:
        self.hidden_size = self.hidden_size // 2

    if self.rnn_type == 'LSTM':
        self.rnn = nn.LSTM(in_channels,
                            self.hidden_size,
                            self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=rnn_dropout,
                            device=self.device)
    elif self.rnn_type == 'GRU':
        self.rnn = nn.GRU(in_channels,
                          self.hidden_size,
                          self.num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=True,
                          dropout=rnn_dropout,
                          device=self.device)
    else:
        raise ValueError("Unsupported RNN type: {}".format(self.rnn_type))

    dropout = params.get('dropout', 0.0)
    self.dropout = nn.Dropout(dropout)
    rnn_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
    self.fc = nn.Linear(rnn_output_size * 3, self.output_size)

  def forward(self, x):
    batch_size = x.shape[0]
    # (batch_size, input_size, seq_length) -> (batch_size, seq_length, input_size)
    x = x.permute(0, 2, 1)

    num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
    h0 = torch.zeros(num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)

    if self.rnn_type == 'LSTM':
        c0 = torch.zeros(num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        rnn_out, (hn, cn) = self.rnn(x, (h0, c0))
    elif self.rnn_type == 'GRU':
        rnn_out, hn = self.rnn(x, h0)
    else:
        raise ValueError("Unsupported RNN type: {}".format(self.rnn_type))

    # Last hidden state
    if self.bidirectional:
        last_hidden_state = torch.cat((hn[-2], hn[-1]), dim=1)
    else:
        last_hidden_state = hn[-1]

    # Concat pooling
    max_pool, _ = torch.max(rnn_out, dim=1)
    avg_pool = torch.mean(rnn_out, dim=1)
    concat_pool = torch.cat((last_hidden_state, max_pool, avg_pool), dim=1)

    out = self.dropout(concat_pool)

    #print(f'Bidi: {self.bidirectional}, hid_size: {self.hidden_size}, last_hidden: {last_hidden_state.shape} out: {out.shape}')
    out = self.fc(out)
    return out
