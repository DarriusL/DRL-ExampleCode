# @Time   : 2023.05.23
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from room.system.onpolicy import OnPolicySystem
from lib import callback, glb_var
import numpy as np

logger = glb_var.get_value('log');

class OffPolicySystem(OnPolicySystem):
    '''System for offpolicy agent'''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);

    def _init_sys(self):
        '''System additional initialization functions'''
        self.rets_mean_valid = [];
        self.total_rewards_valid = [];
        if  self.agent.memory.is_onpolicy:
            logger.error(f'Algorithm [{self.agent.algorithm.name}] is off-policy, while memory [{self.agent.memory}] is on-policy');
            raise callback.CustomException('PolicyConflict');
    
    def _check_train_point(self, epoch):
        '''Check if the conditions for a training session are met'''
        if epoch >= self.agent.train_start_epoch and self.agent.memory.get_stock() >= self.agent.memory.batch_size:
            logger.debug(f'[Epoch: {epoch}] The number of experiences currently stored: [{self.agent.memory.get_stock()}]');
            return True;
        else:
            return False
        
    def _check_valid_point(self, epoch):
        '''Check if the conditions for a validation session are met'''
        #use self._check_train_point(epoch) to ensure the model has been trained
        return self._check_train_point(epoch) and (epoch + 1)%self.cfg['valid']['valid_step'] == 0;
    
    def _explore(self):
        '''Model Exploration Environment

        Notes:
        ------
        Exit differently for training, validation(test falls into verification mode):
        1.training: Force exit when the maximum number of exploration steps is reached.
        2.validation: from the reset environment to the end of the environment.
        '''
        state = self.env.get_state();
        if self.is_training:
            step = 0;
        while True:
            action = self.agent.algorithm.act(state, self.is_training);
            next_state, reward, done, _, _ = self.env.step(action);
            #For PER, use the default omega value (a large constant) to encourage sampling each experience at least once
            self.agent.memory.update(state, action, reward, next_state, done);
            #Check the conditions for exiting the environment
            if self.is_training:
                step += 1;
                if step >= self.agent.expolre_max_step:
                    if done:
                        self.env.reset();
                    break;
                elif done:
                    #not enough data collected
                    self.env.reset();
                    next_state = self.env.get_state();
            elif done:
                #it's test mode
                break;
            state = next_state;

    def _train_epoch(self, epoch):
      for _ in range(self.agent.train_times_per_epoch):
            batch = self.agent.memory.sample();
            for _ in range(self.agent.batch_learn_times_per_train):
                self.agent.algorithm.train_step(batch);

    def train(self):
        return super().train();

    def test(self):
        return super().test()

