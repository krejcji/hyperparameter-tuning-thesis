"""
Load the model to be used for training based on the parameters from configuration file.
"""
def load_model(params):
    match params['model']:
        case 'xresnet1d':
            from xResnet1d import xresnet1d

            model_dropout = params.get('model_dropout', 0.3)
            fc_dropout = params.get('fc_dropout', 0.3)
            original_f_number = params.get('original_f_number', False)
            data_dim = params['data']['input_dim'][0]
            out_dim = params['data']['output_dim']

            model = xresnet1d.xresnet1d34(model_drop_r=model_dropout,
                            original_f_number=original_f_number,
                            fc_drop=fc_dropout, in_ch=data_dim, c_out=out_dim)
        case 'CNN_1D':
            from cnn_net.cnn_1d import CNNNet
            model = CNNNet(params)
        case 'CNN_2D':
            from cnn_net.cnn_2d import CNN2DNet
            model = CNN2DNet(params)
        case 'CNN_2D_simple':
            from cnn_net.cnn_2d_simple import CNN2DSimpleNet
            model = CNN2DSimpleNet(params)
        case 'LSTM':
            from rnn_net.lstm import LSTM
            model = LSTM(params)
        case _:
            raise ValueError(f"Unknown model: {params['model']}")

    return model