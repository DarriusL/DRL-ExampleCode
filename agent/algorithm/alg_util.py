# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
import numpy as np
from lib import util, callback, glb_var

class VarScheduler():
    '''variable scheduler'''
    def __init__(self, var_scheduler_cfg) -> None:
        util.set_attr(self, var_scheduler_cfg);
        self.epoch = 0
        if self.name.lower() == 'linear':
            self.steper = self._linear_scheduler;
        elif self.name.lower() == 'fixed':
            if self.var_start != self.var_end:
                glb_var.get_value('logger').error(f'[Fixed] means stay the same, but the start and end values are different in its configuration');
                raise callback.CustomException('CfgError');
            self.steper = self._fixed_value_scheduler;
        else:
            glb_var.get_value('logger').error(f'Unsupported type [{self.name}]');
            raise callback.CustomException('CfgError');
    
    def step(self):
        return self.steper();

    def _linear_scheduler(self):
        '''linear scheduler'''
        self.epoch += 1;
        if self.epoch < self.star_epoch:
            return self.var_start
        var = self.var_start - (self.var_start - self.var_end)/(self.end_epoch - self.star_epoch) * (self.epoch - self.star_epoch);
        return max(var, self.var_end);

    def _fixed_value_scheduler(self):
        '''fixed value scheduler'''
        return self.var_start;

def cal_returns(rewards, dones, gamma, fast = False):
    '''Compute the returns in the trajectory produced by the Monte Carlo simulation

    Parameters:
    -----------
    rewards:torch.Tensor
        [T]

    dones:torch.Tensor

    gamma:float
    '''
    T = rewards.shape[0];
    if fast:
        g = torch.pow(torch.ones_like(rewards).fill_(gamma), torch.arange(T, device = glb_var.get_value('device')));
        ret_t = rewards * g;
        rets = torch.cumsum(ret_t, dim = -1).flip(dims=[-1])
    else:
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

    Parameters:
    -----------
    action_logit:torch.Tensor
        [..., action_dim]

    Notes:
    ------
    Greedy selection with epsilon probability, random selection with 1-epsilon probability
    '''
    if np.random.random() > epsilon:
        return action_default(action_logit);
    else:
        return action_random(action_logit);

def action_boltzmann(action_logit, tau):
    '''boltzmann policy

    Parameters:
    -----------
    action_logit:torch.Tensor
        [..., action_dim]
    '''
    action_logit = action_logit / tau;
    action_logit = torch.nn.Softmax(dim = -1)(action_logit);
    return action_default(action_logit);