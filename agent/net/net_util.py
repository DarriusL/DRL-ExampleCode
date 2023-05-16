# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from lib import callback, glb_var
from agent import net
from agent.net import *

def get_net(net_cfg, in_dim, out_dim):
    try:
        NetClass = getattr(net, net_cfg['name']);
        return NetClass(net_cfg, in_dim, out_dim);
    except:
        if net_cfg['name'].lower() == 'mlpnet':
            return MLPNet(net_cfg, in_dim, out_dim);
        else:
            glb_var.get_value('logger').error(f'Type of net [{net_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
            callback.CustomException('NetCfgTypeError');

def get_activation_fn(name = 'selu'):
    '''
    Get the activation function

    Parameters:
    ----------
    name: str
        the name of the activation function
        default: 'selu'
    '''
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax', 'selu'];
    if name.lower() == 'sigmoid':
        return torch.nn.Sigmoid();
    elif name.lower() == 'tanh':
        return torch.nn.Tanh();
    elif name.lower() == 'relu':
        return torch.nn.ReLU();
    elif name.lower() == 'leaky_relu':
        return torch.nn.LeakyReLU();
    elif name.lower() == 'elu':
        return torch.nn.ELU();
    elif name.lower() == 'softmax':
        return torch.nn.Softmax();
    elif name.lower() == 'selu':
        return torch.nn.SELU();
    else:
        glb_var.get_value('logger').error(f'Activation function [{name.lower()}] does not support automatic acquisition at the moment,'
                                        f'please replace or add the code yourself.\nSupport list:{activations}');
        callback.CustomException('ActivationCfgNameError');