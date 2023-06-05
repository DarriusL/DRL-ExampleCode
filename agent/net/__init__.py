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

    Returns:
    --------
    net/nets
    '''
    if 'name' not in net_cfg.keys():
        #Configure multiple networks through one cfg
        in_dims, out_dims, net_cfgs = in_dim, out_dim, net_cfg
        assert len(in_dims) == len(out_dims), 'The dimensions of the input and output dims should be the same,' \
            'where the dimensions correspond to the number of networks'
        assert len(net_cfgs.keys()) == len(in_dims), \
            'The number of network configurations should be consistent with the dimension of input and output,' \
            'where the dimension refers to the number of networks'
        nets = [];
        for i, cfg in enumerate(net_cfgs.values()):
            nets.append(get_net(cfg, in_dims[i], out_dims[i]));
        return nets;

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