"""
Load the model to be used for training based on the configuration file.
"""
# TODO: Deal with params
def load_model(config, params = None):
    if config['model']['name'] == 'xresnet1d':
        from xResnet1d import xresnet1d

        model_dropout = config['model']['model_dropout']
        original_f_number = config['model']['original_f_number']
        fc_dropout = config['model']['fc_dropout']

        model = xresnet1d.xresnet1d101(model_drop_r=model_dropout,
                        original_f_number=original_f_number,
                        fc_drop=fc_dropout)
    elif config['model']['name'] == 'CNN':
        from cnn_net.cnn import CNNNet
        model = CNNNet(config)
    elif config['model']['name'] == 'CNN_2D':
        from cnn_net.cnn_2d import CNN2DNet
        model = CNN2DNet(config)
    elif config['model']['name'] == 'CNN_2D_simple':
        from cnn_net.cnn_2d_simple import CNN2DSimpleNet
        model = CNN2DSimpleNet(config, params)
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")

    return model