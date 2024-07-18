"""
This code is responsible for loading the correct model for experiments based
on the configuration.
"""
def load_model(params):
    match params['model']:
        case 'xresnet1d':
            from cnn_net import xresnet1d
            model_dropout = params.get('model_dropout', 0.3)
            fc_dropout = params.get('fc_dropout', 0.25)
            original_f_number = params.get('original_f_number', False)
            data_dim = params['data']['input_dim'][0]
            out_dim = params['data']['output_dim']

            if params['model_size'] == 'xresnet1d18':
                model = xresnet1d.xresnet1d18(model_drop_r=model_dropout,
                                original_f_number=original_f_number,
                                fc_drop=fc_dropout, in_ch=data_dim, c_out=out_dim)
            elif params['model_size'] == 'xresnet1d50':
                model = xresnet1d.xresnet1d50(model_drop_r=model_dropout,
                                original_f_number=original_f_number,
                                fc_drop=fc_dropout, in_ch=data_dim, c_out=out_dim)
            elif params['model_size'] == 'xresnet1d101':
                model = xresnet1d.xresnet1d101(model_drop_r=model_dropout,
                                original_f_number=original_f_number,
                                fc_drop=fc_dropout, in_ch=data_dim, c_out=out_dim)
            elif params['model_size'] == 'xresnet1d34_deep':
                model = xresnet1d.xresnet1d34_deep(model_drop_r=model_dropout,
                                original_f_number=original_f_number,
                                fc_drop=fc_dropout, in_ch=data_dim, c_out=out_dim)
        case 'DenseNet':
            from cnn_net.densenet_torchxrayvision import DenseNet
            inChannels = params['data']['input_dim'][0]
            nClasses = params['data']['output_dim']
            growthRate = 32
            model = DenseNet(growth_rate=growthRate, in_channels=inChannels, num_classes=nClasses)
        case 'CNN_1D':
            from cnn_net.cnn_1d import CNNNet
            model = CNNNet(params)
        case 'CNN_2D':
            from cnn_net.cnn_2d import CNN2DNet
            model = CNN2DNet(params)
        case 'CNN_2D_deep':
            from cnn_net.cnn_2d_deep import CNN2DNet
            model = CNN2DNet(params)
        case 'CNN_2D_simple':
            from cnn_net.cnn_2d_simple import CNN2DSimpleNet
            model = CNN2DSimpleNet(params)
        case 'RNN':
            from rnn_net.rnn import RNN
            model = RNN(params)
        case _:
            raise ValueError(f"Unknown model: {params['model']}")

    return model