from agent.algorithm.reinforce import Reinforce
from agent.algorithm.sarsa import Sarsa
from agent.algorithm.dqn import *
from agent.algorithm.actor_critic import ActorCritic
from agent.algorithm import ppo
from agent.algorithm.acktr import Acktr
from agent import algorithm
from lib import callback, glb_var

__all__ = ['get_alg', 'alg_util'];

logger = glb_var.get_value('log');

def get_alg(alg_cfg):
    '''Obtain the corresponding algorithm object according to the configuration'''
    try:
        AlgClass = getattr(algorithm, alg_cfg['name']);
        alg = AlgClass(alg_cfg);
    except:
        if alg_cfg['name'].lower() == 'reinforce':
            alg = Reinforce(alg_cfg);
        elif alg_cfg['name'].lower() == 'sarsa':
            alg = Sarsa(alg_cfg);
        elif alg_cfg['name'].lower() == 'dqn':
            alg = DQN(alg_cfg);
        elif alg_cfg['name'].lower() == 'targetdqn':
            alg = TargetDQN(alg_cfg);
        elif alg_cfg['name'].lower() == 'doubledqn':
            alg = DoubleDQN(alg_cfg);
        elif alg_cfg['name'].lower() in ['a2c', 'a3c']:
            alg = ActorCritic(alg_cfg);
        elif alg_cfg['name'].lower() == 'ppo_reinforce':
            alg = ppo.Reinforce(alg_cfg);
        elif alg_cfg['name'].lower() in ['ppo_a2c', 'ppo_a3c']:
            alg = ppo.ActorCritic(alg_cfg);
        elif alg_cfg['name'].lower() == 'acktr':
            alg = Acktr(alg_cfg);
        else:
            logger.error(f'Type of algorithm [{alg_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
            raise callback.CustomException('NetCfgTypeError');

        if alg_cfg['name'].lower() in ['a3c', 'ppo_a3c']:
            alg.is_asyn = True;
        else:
            alg.is_asyn = False;

    return alg;