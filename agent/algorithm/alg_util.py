# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
import numpy as np
from lib import util

class VarScheduler():
    '''variable scheduler'''
    def __init__(self, var_scheduler_cfg) -> None:
        util.set_attr(self, var_scheduler_cfg);
        self.epoch = 0
        if self.name.lower() == 'linear':
            self.steper = self._linear_scheduler;
    
    def step(self):
        return self.steper();

    def _linear_scheduler(self):
        '''linear scheduler'''
        self.epoch += 1;
        if self.epoch < self.star_epoch:
            return self.var_start
        var = self.var_start - (self.var_start - self.var_end)/(self.end_epoch - self.star_epoch) * (self.epoch - self.star_epoch);
        return max(var, self.var_end);

def cal_returns(rewards, dones, gamma):
    '''Compute the returns in the trajectory produced by the Monte Carlo simulation

    Parameters:
    -----------
    rewards:torch.Tensor
        [T]

    dones:torch.Tensor

    gamma:float
    '''
    T = rewards.shape[0];
    rets = torch.zeros_like(rewards);
    future_ret = torch.tensor(.0, dtype=rewards.dtype)
    for t in reversed(range(T)):
        future_ret = rewards[t] + gamma * future_ret * (~ dones[t]);
        rets[t] = future_ret;
    return rets;

def cal_nstep_returns(rewards, dones, next_v_pred, gamma, n):
    '''

    '''
    rets = torch.zeros_like(rewards)
    future_ret = next_v_pred
    for t in reversed(range(n)):
        rets[t] = future_ret = rewards[t] + gamma * future_ret * (~ dones[t])
    return rets

def rets_mean_baseline(rets):
    '''Add baselines to returns
    '''
    return rets - rets.mean();

def action_default(action_logit):
    '''default action strategy
    
    Parameters:
    -----------
    action_logit:torch.Tensor
        [..., action_dim]
    '''
    #action probability distribution
    action_pd = torch.distributions.Categorical(logits = action_logit);
    return action_pd.sample();

def action_random(action_logit):
    '''default action strategy

    Parameters:
    -----------
    action_logit:torch.Tensor
        [..., action_dim]
    '''
    return torch.randint(0, action_logit.size()[-1], (1,));

def action_epsilon_greedy(action_logit, epsilon):
    '''epsilon greedy

    '''
    if np.random.random() > epsilon:
        return action_default(action_logit);
    else:
        return action_random(action_logit);