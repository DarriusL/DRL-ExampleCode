# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm import *
from lib import util, callback
from room.system.base import System
import numpy as np
import torch


class OnPolicySystem(System):
    '''System for onpolicy agent'''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);
        self.loss = [];
        self.rets_mean_valid = [];
        self.total_rewards_valid = [];
        if  not self.agent.memory.is_onpolicy:
            self.logger.error(f'Algorithm [{self.agent.algorithm.name}] is on-policy, while memory [{self.agent.memory}] is off-policy');
            raise callback.CustomException('PolicyConflict');

    def _check_train_point(self):
        '''Check if the conditions for a training session are met'''
        if len(self.agent.memory.states) == self.agent.train_exp_size:
            self.logger.debug(f'Current experience length: [{len(self.agent.memory.states)}]\n' 
                f'Training requirements [{self.agent.train_exp_size}].');
            return True;
        else:
            return False;

    def _check_done(self):
        '''Check if the exploration of the current epoch is over'''
        return True if self.env.is_terminated() else False;

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
            action = self.agent.algorithm.act(state, self.env.is_training);
            next_state, reward, done, _, _ = self.env.step(action);
            self.agent.memory.update(state, action, reward, next_state, done);
            #Check the conditions for exiting the environment
            if self.env.is_training:
                if self._check_train_point():
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

    def train(self):
        '''Train the model
        '''
        max_total_rewards = -np.inf;
        valid_not_imporve_cnt = 0;
        for epoch in range(self.agent.max_epoch):
            loss_epoch = [];
            #train
            self.env.train();
            self._explore();
            #start to train
            batch = self.agent.memory.sample();
            self.logger.debug(f'batch data:\n{batch}');
            loss = self.agent.algorithm.train_epoch(batch);
            loss_epoch.append(loss);
            self.loss.append(np.mean(loss_epoch));
            self.logger.debug(f'[train - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}]\n'
                        f'Epoch: [{epoch + 1}/{self.agent.max_epoch}] - train loss: [{self.loss[-1]:.8f}] - '
                        f'lr: [{self.agent.algorithm.optimizer.param_groups[0]["lr"]}]\n');
            if self._check_done():
                self.env.reset();
            self.agent.algorithm.update();
            #valid
            total_rewards = 0;
            rets_mean = 0;
            best_solved = False;
            if (epoch + 1)%self.cfg['valid']['valid_step'] == 0:
                for _ in range(self.cfg['valid']['valid_times']):
                    self.env.eval();
                    self._explore();
                    batch = self.agent.memory.sample();
                    total_rewards += self.env.get_total_reward();
                    rm = alg_util.cal_returns(batch['rewards'], batch['dones'], self.agent.algorithm.gamma).mean().item();
                    rets_mean += rm;
                total_rewards /= self.cfg['valid']['valid_times'];
                rets_mean /= self.cfg['valid']['valid_times'];
                self.total_rewards_valid.append(total_rewards);
                self.rets_mean_valid.append(rets_mean);
                if total_rewards > max_total_rewards:
                    self._save()
                    max_total_rewards = total_rewards;
                    valid_not_imporve_cnt = 0;
                elif total_rewards == max_total_rewards:
                    self._save();
                    valid_not_imporve_cnt += 1;
                else:
                    valid_not_imporve_cnt += 1;
                solved = total_rewards > self.env.solved_total_reward;
                best_solved = max_total_rewards > self.env.solved_total_reward;
                self.logger.info(f'[vaild - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}] - epoch:[{epoch + 1}]\n'
                            f'Mean Returns: [{rets_mean:.3f}] - Total Rewards(now/best): [{total_rewards}/{max_total_rewards}]'
                            f'- solved(now/best): [{solved}/{best_solved}] - not_imporve_cnt: [{valid_not_imporve_cnt}]');
                if (valid_not_imporve_cnt >= self.cfg['valid']['not_improve_finish_step'] and best_solved) or \
                    (max_total_rewards >= self.env.finish_total_reward):
                    break
            self._check_mode();
        self.logger.info(f'Saved Model Information:\nSolved: [{best_solved}] - Mean total rewards: [{max_total_rewards}]'
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
        #plot loss
        util.single_plot(
            np.arange(len(self.loss)), self.loss, 'epoch', 'loss', self.save_path + '/loss.png'
        )

    def test(self):
        self.env.eval()
        self._explore()

