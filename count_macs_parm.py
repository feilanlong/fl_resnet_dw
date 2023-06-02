import torch
from thop import profile,clever_format
def compute_params_flops(logger,cnn, n_channels=3, n_size=32):
    dummy_input = torch.randn(1, n_channels, n_size, n_size)
    macs, params = profile(cnn, inputs=(dummy_input, ), verbose=False) #, custom_ops=custom_ops) 
    macs, params = clever_format([macs, params], "%.3f")
    message = 'macs, params = ' + str(macs) + ', ' + str(params)
    logger.info(message)
