# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var, callback
import torch

class Algorithm():
    '''Abstract Memory class to define the API methods'''
    def __init__(self, algorithm_cfg) -> None:
        util.set_attr(self, algorithm_cfg, except_type = dict);
        self.action_strategy = None;

    def init_net(self):
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def _check_nan(self, loss):
        if torch.isnan(loss):
            self.logger.error('Loss is nan.\nHint:\n(1)Check the loss function;\n'
                              '(2)Checks if the constant used is in the range between [-6.55 x 10^4]~[6.55 x 10^4] when use amp\n'
                              '(3)Not applicable for automatic mixed-precision acceleration.');
            raise callback.CustomException('ValueError');

    def updata(self):
        pass;

    def cal_action_pd(self):
        '''Calculating Action Distribution Parameters'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def act(self, state, is_training):
        '''choose action'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train(self, batch):
        '''Train
        batch is the original data sampled in memory
        '''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;
