# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.net.mlp import *
from agent import net
from agent.net import net_util
from lib import callback

__all__ = ['get_net', 'net_util']
logger = glb_var.get_value('log');

def get_net(net_cfg, in_dim, out_dim):
    '''
    Obtain the corresponding network according to the configuration

    Parameters:
    ----------
    net_cfg:dict

    in_dim

    out_dim
    '''
    try:
        NetClass = getattr(net, net_cfg['name']);
        return NetClass(net_cfg, in_dim, out_dim);
    except:
        if net_cfg['name'].lower() == 'mlpnet':
            return MLPNet(net_cfg, in_dim, out_dim);
        elif net_cfg['name'].lower() == 'sharedmlpnet':
            return SharedMLPNet(net_cfg, in_dim, out_dim);
        else:
            logger.error(f'Type of net [{net_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
            raise callback.CustomException('NetCfgTypeError');