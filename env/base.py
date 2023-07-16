# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import util, glb_var, callback
import gym

logger = glb_var.get_value('log');

def _make_env(env_cfg):
    if env_cfg['name'].lower() == 'cartpole':
        if glb_var.get_value('mode') == 'train':
            return gym.make("CartPole-v1");
        else:
            return gym.make("CartPole-v1", render_mode="human");
    elif env_cfg['name'].lower() == 'mountaincar':
        if glb_var.get_value('mode') == 'train':
            return gym.make("MountainCar-v0").env;
        else:
            return gym.make("MountainCar-v0", render_mode="human").env;
    elif env_cfg['name'].lower() == 'pong':
        if glb_var.get_value('mode') == 'train':
            env = gym.make('Pong-v4');
        else:
            env = gym.make('Pong-v4', render_mode="human");
        return gym.wrappers.TimeLimit(env, max_episode_steps=1000);
    else:
        logger.error(f'Type of env [{env_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');

class Env():
    '''Abstract Env class to define the API methods'''
    def __init__(self, env_cfg) -> None:
        util.set_attr(self, env_cfg);
        self.is_training = True;

    def set_seed(self, seed):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train(self):
        '''train mode'''
        self.is_training = True;
    
    def valid(self):
        '''validation mode'''
        self.is_training = False;

    def get_state_and_action_dim(self):
        '''Get the choice space of environment state dimensions and actions'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def get_state(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def reset(self):
        '''reset environment'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def step(self, action):
        '''change the environment'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;