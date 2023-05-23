# @Time   : 2023.05.23
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from room.system.base import System
from lib import callback
import torch

class OffPolicySystem(System):
    '''System for offpolicy agent'''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);
        self.loss = [];
        self.rets_mean_valid = [];
        self.total_rewards_valid = [];
        if  self.agent.memory.is_onpolicy:
            self.logger.error(f'Algorithm [{self.agent.algorithm.name}] is off-policy, while memory [{self.agent.memory}] is on-policy');
            raise callback.CustomException('PolicyConflict');

    def _save(self):
        '''Save model
        '''
        torch.save(self.agent.algorithm, self.cfg['model_path']);

    def train(self):
        ''''''
