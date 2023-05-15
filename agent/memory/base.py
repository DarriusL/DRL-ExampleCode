# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var

class Memory():
    '''Abstract Memory class to define the API methods'''

    def __init__(self, memory_cfg) -> None:
        util.set_attr(self, memory_cfg);
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones'];

    def reset(self):
        '''Reset experience memory memory'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def update(self, state, action, reward, next_state, done):
        '''Add experience to memory'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def sample(self):
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;