# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var, callback
import torch

logger = glb_var.get_value('log')

class Algorithm():
    '''Abstract Memory class to define the API methods'''
    def __init__(self, algorithm_cfg) -> None:
        util.set_attr(self, algorithm_cfg, except_type = dict);
        glb_var.get_value('var_reporter').add('Gamma', self.gamma);
        self.action_strategy = None;
        #Tags for asynchronous parallel computing
        self.is_asyn = False;

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def _check_nan(self, loss):
        if torch.isnan(loss):
            logger.error('Loss is nan.\nHint:\n(1)Check the loss function;\n'
                              '(2)Checks if the constant used is in the range between [-6.55 x 10^4]~[6.55 x 10^4] when use amp\n'
                              '(3)Not applicable for automatic mixed-precision acceleration.');
            raise callback.CustomException('ValueError');

    def update(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def _cal_action_pd(self, state):
        '''Calculating Action Distribution Parameters'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def act(self, state, is_training):
        '''choose action'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train_step(self, batch):
        '''Train
        batch is the original data sampled in memory
        '''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;
