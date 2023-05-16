# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import gym
from lib import glb_var, callback

def make_env(env_cfg):
    if env_cfg['name'].lower() in 'cartpole':
        return gym.make("CartPole-v1");
    else:
        glb_var.get_value('logger').error(f'Type of env [{env_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');