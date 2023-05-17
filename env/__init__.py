# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from env.openai_gym import OpenaiEnv

__all__ = ['get_env'];

def get_env(env_cfg):
    return OpenaiEnv(env_cfg);
