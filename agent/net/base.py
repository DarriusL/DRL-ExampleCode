# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from lib import glb_var, util

logger = glb_var.get_value('log');

class Net(torch.nn.Module):
    '''Abstract Net class to define the API methods
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__();
        util.set_attr(self, net_cfg);

    def _init_para(self, module):
        '''Network parameter initialization'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;
    
    def forward(self):
        '''network forward pass'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;