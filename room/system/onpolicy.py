# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm import *
from lib import util, callback, glb_var
from room.system.base import System
import numpy as np
import torch, copy, time

logger = glb_var.get_value('log');

class OnPolicySystem(System):
    '''System for onpolicy agent
    
    Notes:
    ------
    The following algorithm is applicable to the system.
    `Algorithm:Reinforce,SARSA,A2C,Reinforce(PPO), A2C(PPO)
    `Memory:OnPolicyMemory,OnPolicyBatchMemory

    *At the same time, 
    *the Off-Policy algorithm can also use the system in combination with the above memory, 
    *but the training efficiency will be lower
    '''
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
        logger.debug(f'Current experience length: [{len(self.agent.memory.states)}]\n' 
            f'Training requirements [{self.agent.train_exp_size * self.agent.explore_times_per_train}].');
        return len(self.agent.memory.states) == self.agent.train_exp_size*self.agent.explore_times_per_train;

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
                    next_state = self.env.reset();
            elif done:
                #it's test mode
                break;
            state = next_state;

    def _save(self):
        '''Save the algorithm model
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
        #valid
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
        #check save point
        if total_rewards > self.max_total_rewards:
            #better, save
            self._save()
            self.max_total_rewards = total_rewards;
            self.valid_not_imporve_cnt = 0;
        elif total_rewards == self.max_total_rewards:
            if max(self.rets_mean_valid) == rets_mean: 
                #better, save
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
            #collect experiences
            self._explore();
            #check for off policy algorithm
            if self._check_train_point(epoch):
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

class OnPolicyAsynSubSystem(OnPolicySystem):
    '''Asynchronous Parallel System Subsystem

    Notes:
    ------
    A system can only be used for one of the training or simulation.
    '''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);

    def init_sys(self, rank, shared_alg, optimzer):
        '''System additional initialization functions'''
        self.env.set_seed(rank);
        torch.manual_seed(rank);
        self.rank = rank;
        cfg = self.cfg;
        state_dim, action_dim = self.env.get_state_and_action_dim();
        self.agent.algorithm.init_net(
            cfg['agent_cfg']['net_cfg'],
            None,
            cfg['agent_cfg']['lr_schedule_cfg'],
            state_dim,
            action_dim,
            cfg['agent_cfg']['max_epoch'],
            optimzer
        );
        self.agent.algorithm.set_shared_net(shared_alg);
    
    def train(self, lock, stop_event, cnt, end_cnt, rank, shared_alg, optimzer):
        '''Train process'''
        self.init_sys(rank, shared_alg, optimzer);
        for epoch in range(self.agent.max_epoch):
            if stop_event.is_set():
                break;
            with lock:
                self.agent.algorithm.load_sharednet();
            #train mode
            self.train_mode();
            #collect experiences
            self._explore();
            #start to train
            self._train_epoch(epoch);
            #algorithm update
            self.agent.algorithm.update();
            with lock:
                cnt.value += 1;
        logger.info(f'Process {self.rank} end.');
        with lock:
            end_cnt.value += 1;

    def valid(self, lock, stop_event, cnt, end_cnt, rank, shared_alg, optimzer):
        '''Valid process'''
        self.init_sys(rank, shared_alg, optimzer)
        while True:
            #Here, take valid_step as the starting point
            if cnt.value > self.cfg['valid']['valid_step']:
                with lock:
                    self.agent.algorithm.load_sharednet();
                    cnt_value = cnt.value;
                if self._valid_epoch(cnt_value):
                    stop_event.set();
                    break;
                if end_cnt == self.rank:
                    break;
                time.sleep(60);
        logger.info(f'Saved Model Information:\nSolved: [{self.best_solved}] - Mean total rewards: [{self.max_total_rewards}]'
                    f'\nSaved path:{self.save_path}');
        if end_cnt != self.rank:
            #plot rets
            util.single_plot(
                np.arange(len(self.rets_mean_valid)) + 1,
                self.rets_mean_valid,
                'valid_times', 'mean_rets', self.save_path + '/mean_rets.png');
            #plot total rewards
            util.single_plot(
                np.arange(len(self.total_rewards_valid)) + 1,
                self.total_rewards_valid,
                'valid_times', 'rewards', self.save_path + '/rewards.png');
        

class OnPolicyAsynSystem(OnPolicySystem):
    '''Asynchronous Parallel System

    Refrence:
    ---------
    [1]https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py
    [2]https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py
    [3]https://zhuanlan.zhihu.com/p/346205754
    '''
    def __init__(self, cfg, algorithm, env) -> None:
        super().__init__(cfg, algorithm, env);

    def train(self):
        '''Train the model.'''
        subtrainsystems = [OnPolicyAsynSubSystem(
            copy.deepcopy(self.cfg),
            get_alg(self.cfg['agent_cfg']['algorithm_cfg']),
            copy.deepcopy(self.env)
        ) for _ in range(self.agent.asyn_num + 1)];
        optimizer = self.agent.algorithm.get_optimizer();
        self.agent.algorithm.share_memory();
        subvalidsystem = copy.deepcopy(subtrainsystems[-1]);
        del subtrainsystems[-1];
        cnt = torch.multiprocessing.Value('i', 0);
        end_cnt = torch.multiprocessing.Value('i', 0);
        lock = torch.multiprocessing.Lock();
        glb_var.set_value('lock', lock);
        stop_event = torch.multiprocessing.Event();
        processes = [];
        for rank, sys in enumerate(subtrainsystems):
            p = torch.multiprocessing.Process(
                target = sys.train, 
                args = (lock, stop_event, cnt, end_cnt, rank, self.agent.algorithm, optimizer)
                );
            p.start();
            processes.append(p);
        p_valid = torch.multiprocessing.Process(
            target = subvalidsystem.valid, 
            args = (lock, stop_event, cnt, end_cnt, rank + 1, self.agent.algorithm, optimizer)
            );
        p_valid.start();
        processes.append(p_valid);
        for p in processes:
            p.join()
