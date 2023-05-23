# @Time   : 2023.05.23
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from room.system.onpolicy import OnPolicySystem
from lib import callback
import numpy as np

class OffPolicySystem(OnPolicySystem):
    '''System for offpolicy agent'''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);

    def _init_sys(self):
        '''System additional initialization functions'''
        self.loss = [];
        self.rets_mean_valid = [];
        self.total_rewards_valid = [];
        if  self.agent.memory.is_onpolicy:
            self.logger.error(f'Algorithm [{self.agent.algorithm.name}] is off-policy, while memory [{self.agent.memory}] is on-policy');
            raise callback.CustomException('PolicyConflict');
    
    def _check_train_point(self, epoch):
        '''Check if the conditions for a training session are met'''
        if epoch >= self.agent.train_start_epoch and self.agent.memory.get_stock() >= self.agent.memory.batch_size:
            self.logger.debug(f'[Epoch: {epoch}] The number of experiences currently stored: [{self.agent.memory.get_stock()}]');
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
        '''Model training per epoch'''
        #start to train
        for _ in range(self.agent.train_times_per_epoch):
            loss_epoch = [];
            batch = self.agent.memory.sample();
            self.logger.debug(f'batch: {batch}');
            for _ in range(self.agent.batch_learn_times_per_train):
                loss_epoch.append(self.agent.algorithm.train_step(batch));
        self.loss.append(np.mean(loss_epoch));
        self.logger.debug(f'[train - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}]\n'
                    f'Epoch: [{epoch + 1}/{self.agent.max_epoch}] - train loss: [{self.loss[-1]:.8f}] - '
                    f'lr: [{self.agent.algorithm.optimizer.param_groups[0]["lr"]}]\n');

    def train(self):
        return super().train();

    def test(self):
        return super().test()

