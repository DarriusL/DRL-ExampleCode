# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var

logger = glb_var.get_value('log');

class Memory():
    '''Abstract Memory class to define the API methods
    
    Notes:
    ------
    1. train(), valid() are used to switch modes, 
       train mode is used to store data during training, 
       and valid mode is used to store data during verification.
    2. valid mode uses on-policy memory.
    3. Due to its particularity (it will be cleared after each sampling), the on-policy memory can be used directly
    '''
    def __init__(self, memory_cfg) -> None:
        util.set_attr(self, memory_cfg, except_type = dict);
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones'];
        self.is_onpolicy = None;

    def _batch_to_tensor(self):
        '''Convert a batch to a format for torch training'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def get_stock(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def valid(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def reset(self):
        '''Reset experience memory memory'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def update(self, state, action, reward, next_state, done):
        '''Add experience to memory'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def sample(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;