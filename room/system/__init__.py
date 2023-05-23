# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm import *
from env import *
from room.system.onpolicy import *
from room.system.offpolicy import *
from lib import glb_var

__all__ = ['get_system']

def get_system(cfg):
    '''Generate systems according to on-policy and off-policy'''
    #env
    env = get_env(cfg['env']);
    #action_dim : numbers of all actions
    state_dim, action_dim = env.get_state_and_action_dim();
    #algorithm
    if glb_var.get_value('mode') == 'test':
        algorithm = torch.load(cfg['model_path']);
    elif glb_var.get_value('mode') == 'train':
        algorithm = get_alg(cfg['agent_cfg']['algorithm_cfg']);
        #Initialize the agent's network
        algorithm.init_net(
            cfg['agent_cfg']['net_cfg'],
            cfg['agent_cfg']['optimizer_cfg'],
            cfg['agent_cfg']['lr_schedule_cfg'],
            cfg['agent_cfg']['algorithm_cfg']['var_schedule_cfg'],
            state_dim,
            action_dim,
            cfg['agent_cfg']['max_epoch']
        );
    #Generating systems based on on-policy and off-policy algorithms
    if algorithm.is_onpolicy:
        return OnPolicySystem(cfg, algorithm, env);
    elif not algorithm.is_onpolicy:
        return OffPolicySystem(cfg, algorithm, env);
    else:
        pass
