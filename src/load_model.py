"""
Load the model to be used for training based on the parameters from configuration file.
"""
def load_model(params):
    if params['model']['name'] == 'xresnet1d':
        from xResnet1d import xresnet1d

        model_dropout = params['model']['model_dropout']
        original_f_number = params['model']['original_f_number']
        fc_dropout = params['model']['fc_dropout']

        model = xresnet1d.xresnet1d101(model_drop_r=model_dropout,
                        original_f_number=original_f_number,
                        fc_drop=fc_dropout)
    elif params['model']['name'] == 'CNN':
        from cnn_net.cnn import CNNNet
        model = CNNNet(params)
    elif params['model']['name'] == 'CNN_2D':
        from cnn_net.cnn_2d import CNN2DNet
        model = CNN2DNet(params)
    elif params['model']['name'] == 'CNN_2D_simple':
        from cnn_net.cnn_2d_simple import CNN2DSimpleNet
        model = CNN2DSimpleNet(params)
    else:
        raise ValueError(f"Unknown model: {params['model']['name']}")

    return model