from agent.algorithm.reinforce import Reinforce
from agent.algorithm.sarsa import Sarsa
from agent.algorithm.dqn import *
from agent import algorithm
from lib import callback, glb_var

__all__ = ['get_alg', 'alg_util'];

def get_alg(alg_cfg):
    '''Obtain the corresponding algorithm object according to the configuration'''
    try:
        AlgClass = getattr(algorithm, alg_cfg['name']);
        return AlgClass(alg_cfg);
    except:
        if alg_cfg['name'].lower() == 'reinforce':
            return Reinforce(alg_cfg);
        elif alg_cfg['name'].lower() == 'sarsa':
            return Sarsa(alg_cfg);
        elif alg_cfg['name'].lower() == 'classicdqn':
            return ClassicDQN(alg_cfg);
        else:
            glb_var.get_value('logger').error(f'Type of algorithm [{alg_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
            raise callback.CustomException('NetCfgTypeError');