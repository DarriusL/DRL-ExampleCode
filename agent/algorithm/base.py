# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var

class Algorithm():
    '''Abstract Memory class to define the API methods'''
    def __init__(self, algorithm_cfg) -> None:
        util.set_attr(self, algorithm_cfg);

    def _init_net(self):
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;
    
    def cal_action_pd(self):
        '''Calculating Action Distribution Parameters'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def act(self, state):
        '''choose action'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train(self, batch):
        '''Train
        batch is the original data sampled in memory
        '''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;
