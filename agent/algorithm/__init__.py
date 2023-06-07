from agent.algorithm.reinforce import Reinforce
from agent.algorithm.sarsa import Sarsa
from agent.algorithm.dqn import *
from agent.algorithm.actor_critic import ActorCritic
from agent.algorithm import ppo
from agent import algorithm
from lib import callback, glb_var

__all__ = ['get_alg', 'alg_util'];

logger = glb_var.get_value('log');

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
        elif alg_cfg['name'].lower() == 'dqn':
            return DQN(alg_cfg);
        elif alg_cfg['name'].lower() == 'targetdqn':
            return TargetDQN(alg_cfg);
        elif alg_cfg['name'].lower() == 'doubledqn':
            return DoubleDQN(alg_cfg);
        elif alg_cfg['name'].lower() in ['a2c']:
            return ActorCritic(alg_cfg);
        elif alg_cfg['name'].lower() == 'ppo_reinforce':
            return ppo.Reinforce(alg_cfg);
        else:
            logger.error(f'Type of algorithm [{alg_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
            raise callback.CustomException('NetCfgTypeError');