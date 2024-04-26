"""
Load the model to be used for training based on the parameters from configuration file.
"""
def load_model(params):
    match params['model']:
        case 'xresnet1d':
            from xResnet1d import xresnet1d

            # model_dropout = params['model_dropout']
            # original_f_number = params['original_f_number']
            # fc_dropout = params['fc_dropout']

            model_dropout = 0.2
            original_f_number = True
            fc_dropout = 0.3


            model = xresnet1d.xresnet1d34(model_drop_r=model_dropout,
                            original_f_number=original_f_number,
                            fc_drop=fc_dropout)
        case 'CNN_1D':
            from cnn_net.cnn_1d import CNNNet
            model = CNNNet(params)
        case 'CNN_2D':
            from cnn_net.cnn_2d import CNN2DNet
            model = CNN2DNet(params)
        case 'CNN_2D_simple':
            from cnn_net.cnn_2d_simple import CNN2DSimpleNet
            model = CNN2DSimpleNet(params)
        case _:
            raise ValueError(f"Unknown model: {params['model']}")

    return model