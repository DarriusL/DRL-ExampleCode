# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory import *
from agent.algorithm import *
from agent.net import *
from dataclasses import dataclass
from lib import glb_var, util, json_util
import matplotlib.pyplot as plt
from env import *
import numpy as np
import torch, os

@dataclass
class Agent:
    memory:None
    algorithm:None

class System():
    ''''''
    def __init__(self, cfg) -> None:
        self.cfg = cfg;
        self.loss = [];
        self.rets_mean = [];
        self.rets_mean_valid = [];
        self.logger = glb_var.get_value('logger');
        self.env = get_env(cfg['env']);
        #action_dim : numbers of all actions
        state_dim, action_dim = self.env.get_state_and_action_dim();
        if glb_var.get_value('mode') == 'test':
            self.save_path, _ = os.path.split(self.cfg['model_path']);
            algorithm = torch.load(self.cfg['model_path']);
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
            self.save_path = glb_var.get_value('save_dir') + f'/{algorithm.name.lower()}/{self.env.name.lower()}/{util.get_date("_")}';
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path);
            self.cfg['model_path'] = self.save_path + '/alg.model';
            json_util.jsonsave(self.cfg, self.save_path + '/config.json'); 

        memory = get_memory(cfg['agent_cfg']['memory_cfg']);
        self.agent = Agent(memory, algorithm);
        util.set_attr(self.agent, cfg['agent_cfg'], except_type = dict)
        self.env.reset();


    def _check_train_point(self):
        '''Check if the conditions for a training session are met'''

        self.logger.debug(f'Current experience length: [{len(self.agent.memory.states)}]\n' 
                        f'Training requirements [{self.agent.train_exp_size}].');
        return True if len(self.agent.memory.states) == self.agent.train_exp_size else False;

    def _check_done(self):
        '''Check if the exploration of the current epoch is over'''
        return True if self.env.is_terminated() else False;

    def _explore(self):
        '''Model Exploration Environment
        '''
        state = self.env.get_state();
        while True:
            action = self.agent.algorithm.act(state, self.env.is_training);
            next_state, reward, done, _, _ = self.env.step(action);
            self.agent.memory.update(state, action, reward, next_state, done);
            if done or (self.env.is_training and self._check_train_point()):
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
            rets_mean_epoch = [];
            while True:
                #train
                self.env.train();
                self._explore();
                if self._check_train_point():
                    #start to train
                    batch = self.agent.algorithm.batch_to_tensor(self.agent.memory.sample());
                    loss, rets_mean = self.agent.algorithm.train_epoch(batch);
                    loss_epoch.append(loss);
                    rets_mean_epoch.append(rets_mean);
                if self._check_done():
                    self.loss.append(np.mean(loss_epoch));
                    self.rets_mean.append(np.mean(rets_mean_epoch));
                    total_rewards = self.env.get_total_reward();
                    solved = total_rewards > self.env.solved_total_reward;
                    self.logger.info(f'[train - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}]\n'
                                f'Epoch: [{epoch + 1}/{self.agent.max_epoch}] - train loss: [{self.loss[-1]:.8f}] - '
                                f'lr: [{self.agent.algorithm.optimizer.param_groups[0]["lr"]}]\n'
                                f'Mean Returns: [{self.rets_mean[-1]:.3f}] - Total Rewards: [{total_rewards}] - solved: [{solved}]');
                    self.env.reset();
                    break;

            #valid
            total_rewards = 0;
            rets_mean = 0;
            best_solved = False;
            if (epoch + 1)%self.cfg['valid']['valid_step'] == 0:
                for _ in range(self.cfg['valid']['valid_times']):
                    self.env.eval();
                    self._explore();
                    batch = self.agent.algorithm.batch_to_tensor(self.agent.memory.sample());
                    total_rewards += self.env.get_total_reward();
                    _, rm = self.agent.algorithm.cal_rets(batch);
                    
                    rets_mean += rm;
                total_rewards /= self.cfg['valid']['valid_times'];
                rets_mean /= self.cfg['valid']['valid_times'];
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
                self.logger.info(f'[vaild - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}]\n'
                            f'Mean Returns: [{rets_mean:.3f}] - Total Rewards(now/best): [{total_rewards}/{max_total_rewards}]'
                            f'- solved(now/best): [{solved}/{best_solved}] - not_imporve_cnt: [{valid_not_imporve_cnt}]');
        
                if (valid_not_imporve_cnt >= self.cfg['valid']['not_improve_finish_step'] and best_solved) or \
                    (max_total_rewards >= self.env.finish_total_reward):
                    break
        self.logger.info(f'Saved Model Information:\nSolved: [{best_solved}] - Mean total rewards: [{max_total_rewards}]');
        plt.figure(figsize = (10, 6));
        plt.plot(np.arange(0, len(self.rets_mean)) + 1, self.rets_mean, label = 'train');
        plt.plot(
            np.arange(self.cfg['valid']['valid_step'] - 1, len(self.rets_mean), self.cfg['valid']['valid_step']) + 1, 
            self.rets_mean_valid, 
            label = 'valid');
        plt.xlabel('epoch');
        plt.ylabel('mean_rets');
        plt.yscale('log');
        plt.legend(loc='lower right')
        plt.savefig(self.save_path + '/loss.png', dpi = 400);

    def test(self):
        self._explore(train = True)

