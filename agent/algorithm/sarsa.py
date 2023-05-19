# @Time   : 2023.05.19
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.net import *
from agent.algorithm.base import Algorithm
from agent.algorithm import alg_util
from lib import glb_var
import torch
import numpy as np

class Sarsa(Algorithm):
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.action_strategy = alg_util.action_epsilon_greedy;
        

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, var_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        self.q_net = get_net(net_cfg, in_dim, out_dim).to(glb_var.get_value('device'));
        self.optimizer = net_util.get_optimizer(optim_cfg, self.pi);
        #if None, then do not use
        self.lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, self.optimizer, max_epoch);
        self.var_schedule = alg_util.VarScheduler(var_schedule_cfg, max_epoch);
        self.epsilon = self.var_schedule.var_start;

    def batch_to_tensor(self, batch):
        '''Convert a batch to a format for torch training
        batch['states]:[T, in_dim]
        batch['actions']:[T]
        batch['rewards']:[T]
        batch['next_states']:[T, in_dim]
        '''
        batch['next_actions'] = np.zeros_like(batch['actions'])
        batch['next_actions'][:-1] = batch['actions'][1:]
        for key in batch.keys():
            batch[key] = torch.from_numpy(np.array(batch[key])).to(glb_var.get_value('device'));
        return batch;

    def cal_action_pd(self, state):
        '''
        Action distribution probability in the input state

        Parameters:
        ----------
        state:torch.Tensor
        '''
        #x:[..., in_dim]
        #return [..., out_dim]
        return self.q_net(state);

    @torch.no_grad()
    def act(self, state, is_training):
        '''take action on input state

        Parameters:
        -----------
        state:numpy.ndarray

        is_training:bool

        Returns:
        --------
        action
        '''
        #the logarithm of the action probability distribution
        action_logit = self.pi(torch.from_numpy(state).to(torch.float32).to(glb_var.get_value('device')));
        if is_training:
            action = self.action_strategy(action_logit);
        else:
            action = alg_util.action_default(action_logit);
        return action.cpu().item();

    def cal_rets(self, batch):
        pass;

    def cal_loss(self, batch):
        '''Calculate MSELoss for SARSA'''
        #[T, out_dim]
        q_preds_table = self.q_net(batch['states']);
        #[T, out_dim]
        with torch.no_grad():
            next_q_preds_table = self.q_net(batch['next_states']);
        #[T]
        q_pred = q_preds_table.gather(-1, batch['actions'].unsqueeze(-1)).squeeze(-1);
        #[T]
        next_q_preds = next_q_preds_table.gather(-1, batch['next_actions'].unsqueeze(-1)).squeeze(-1);
        q_tar_preds = batch['rewards'] + self.gamma * next_q_preds * (~ batch['dones']);
        return torch.nn.MSELoss()(q_pred, q_tar_preds);

    def train_epoch(self, batch):
        '''training network

        Parameters:
        -----------
        batch:dict
            Convert through batch_to_tensor before passing in
        '''
        #[T, out_dim]
        action_batch_logits = self.cal_action_pd_batch(batch);
        loss = self.cal_loss(action_batch_logits, batch);
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        self.optimizer.zero_grad();
        self._check_nan(loss);
        loss.backward();
        self.optimizer.step();
        if self.lr_schedule is not None:
            self.lr_schedule.step();
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        return loss.item(), rets_mean;