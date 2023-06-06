# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm import *
from lib import util, callback, glb_var
from room.system.base import System
import numpy as np
import torch

logger = glb_var.get_value('log');

class OnPolicySystem(System):
    '''System for onpolicy agent'''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);
        self.rets_mean_valid = [];
        self.total_rewards_valid = [];
        self.max_total_rewards = -np.inf;
        self.valid_not_imporve_cnt = 0;
        self.best_solved = False;
        self._init_sys();

    def _init_sys(self):
        '''System additional initialization functions'''
        if  not self.agent.memory.is_onpolicy:
            logger.error(f'Algorithm [{self.agent.algorithm.name}] is on-policy, while memory [{self.agent.memory}] is off-policy');
            raise callback.CustomException('PolicyConflict');

    def _check_train_point(self, epoch):
        '''Check if the conditions for a training session are met'''
        if len(self.agent.memory.states) == self.agent.train_exp_size*self.agent.explore_times_per_train:
            logger.debug(f'Current experience length: [{len(self.agent.memory.states)}]\n' 
                f'Training requirements [{self.agent.train_exp_size * self.agent.explore_times_per_train}].');
            return True;
        else:
            return False;

    def _check_valid_point(self, epoch):
        '''Check if the conditions for a validation session are met'''
        return (epoch + 1)%self.cfg['valid']['valid_step'] == 0;

    def _explore(self):
        '''Model Exploration Environment

        Notes:
        ------
        Exit differently for training, validation(test falls into verification mode):
        1.training: the data length requirement must be met to force exit.
        2.validation: from the reset environment to the end of the environment.
        '''
        state = self.env.get_state();
        while True:
            action = self.agent.algorithm.act(state, self.is_training);
            next_state, reward, done, _, _ = self.env.step(action);
            self.agent.memory.update(state, action, reward, next_state, done);
            #Check the conditions for exiting the environment
            if self.is_training:
                if self._check_train_point(None):
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

    def _save(self):
        '''Save model
        '''
        torch.save(self.agent.algorithm, self.cfg['model_path']);
    
    def _train_epoch(self, epoch):
        '''Model training per epoch
        '''
        batch = self.agent.memory.sample();
        for _ in range(self.agent.batch_learn_times_per_train):
            self.agent.algorithm.train_step(batch);

    def _valid_epoch(self, epoch):
        '''Model validation per epoch'''
        total_rewards = 0;
        rets_mean = 0;
        for _ in range(self.cfg['valid']['valid_times']):
            self.valid_mode();
            self._explore();
            batch = self.agent.memory.sample();
            total_rewards += self.env.get_total_reward();
            rm = alg_util.cal_returns(batch['rewards'], batch['dones'], self.agent.algorithm.gamma, fast = True).mean().item();
            rets_mean += rm;
        total_rewards /= self.cfg['valid']['valid_times'];
        rets_mean /= self.cfg['valid']['valid_times'];
        self.total_rewards_valid.append(total_rewards);
        self.rets_mean_valid.append(rets_mean);
        if total_rewards > self.max_total_rewards:
            self._save()
            self.max_total_rewards = total_rewards;
            self.valid_not_imporve_cnt = 0;
        elif total_rewards == self.max_total_rewards:
            if max(self.rets_mean_valid) == rets_mean: 
                self._save();
            self.valid_not_imporve_cnt += 1;
        else:
            self.valid_not_imporve_cnt += 1;
        solved = total_rewards > self.env.solved_total_reward;
        self.best_solved = self.max_total_rewards > self.env.solved_total_reward;
        content_head = f'[vaild - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}] - Epoch:[{epoch + 1}]\n'\
                        f'Mean Returns: [{rets_mean:.3f}] - Total Rewards(now/best): [{total_rewards}/{self.max_total_rewards}]'\
                        f'- solved(now/best): [{solved}/{self.best_solved}] - not_imporve_cnt: [{self.valid_not_imporve_cnt}]';
        glb_var.get_value('var_reporter').report(logger.info, content_head, 4);
        return (self.valid_not_imporve_cnt >= self.cfg['valid']['not_improve_finish_step'] and self.best_solved) or \
            (self.max_total_rewards >= self.env.finish_total_reward);

    def train(self):
        '''The main function to train the model

        Notes:
        ------
        If you need to reuse the code, you only need to rewrite 
        [_check_train_point, _check_valid_point, _explore, _train_epoch, _valid_epoch]. 
        In general, the verification does not need to be rewritten.
        '''
        for epoch in range(self.agent.max_epoch):
            #train mode
            self.train_mode();
            self._explore();
            #start to train
            self._train_epoch(epoch);
            #algorithm update
            self.agent.algorithm.update();
            #valid mode
            if self._check_valid_point(epoch):
                if self._valid_epoch(epoch):
                    break;
            self._check_mode();
        logger.info(f'Saved Model Information:\nSolved: [{self.best_solved}] - Mean total rewards: [{self.max_total_rewards}]'
                         f'\nSaved path:{self.save_path}');
        #plot rets
        util.single_plot(
            np.arange(self.cfg['valid']['valid_step'] - 1, epoch + 1, self.cfg['valid']['valid_step']) + 1,
            self.rets_mean_valid,
            'epoch', 'mean_rets', self.save_path + '/mean_rets.png');
        #plot total rewards
        util.single_plot(
            np.arange(self.cfg['valid']['valid_step'] - 1, epoch + 1, self.cfg['valid']['valid_step']) + 1,
            self.total_rewards_valid,
            'epoch', 'rewards', self.save_path + '/rewards.png');


    def test(self):
        '''Test the model'''
        self.valid_mode();
        self._explore();
        batch = self.agent.memory.sample();
        total_rewards = self.env.get_total_reward();
        rets_mean = alg_util.cal_returns(batch['rewards'], batch['dones'], self.agent.algorithm.gamma, fast = True).mean().item();
        logger.info(f'[test - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}]\n'\
                    f'Mean Returns: [{rets_mean:.3f}] - Total Rewards: [{total_rewards}]');

