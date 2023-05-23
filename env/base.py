# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var, callback
import gym

def _make_env(env_cfg):
    if env_cfg['name'].lower() in 'cartpole':
        if glb_var.get_value('mode') == 'train':
            return gym.make("CartPole-v1");
        else:
            return gym.make("CartPole-v1", render_mode="human");
    else:
        glb_var.get_value('logger').error(f'Type of env [{env_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');

class Env():
    '''Abstract Env class to define the API methods'''
    def __init__(self, env_cfg) -> None:
        util.set_attr(self, env_cfg);
        self.is_training = True;
    
    def train(self):
        '''train mode'''
        self.is_training = True;
    
    def valid(self):
        '''validation mode'''
        self.is_training = False;

    def get_state_and_action_dim(self):
        '''Get the choice space of environment state dimensions and actions'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def get_state(self):
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def reset(self):
        '''reset environment'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def step(self, action):
        '''change the environment'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;