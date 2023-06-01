# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
import numpy as np
from lib import util, callback, glb_var

logger = glb_var.get_value('log')

class VarScheduler():
    '''variable scheduler'''
    def __init__(self, var_scheduler_cfg) -> None:
        util.set_attr(self, var_scheduler_cfg);
        self.epoch = 0
        if self.name.lower() == 'linear':
            self.steper = self._linear_scheduler;
        elif self.name.lower() == 'fixed':
            if self.var_start != self.var_end:
                logger.error(f'[Fixed] means stay the same, but the start and end values are different in its configuration');
                raise callback.CustomException('CfgError');
            self.steper = self._fixed_value_scheduler;
        else:
            logger.error(f'Unsupported type [{self.name}]');
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
        [T], bool

    gamma:float

    fast:bool, optional
        If enabled, use more efficient fast calculations.
         default:False
    
    Notes:
    ------
    about [fast]:Problems with [fast] if experience comes from different tracks.
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
    '''Compute the returns in n steps

    Parameters:
    -----------
    rewards:torch.Tensor
    [T]
    
    dones:torch.Tensor
    [T], bool
    
    gamma:float

    next_v_pred:torch.Tensor
    [T]
    '''
    rets = torch.zeros_like(rewards)
    future_ret = next_v_pred
    for t in reversed(range(n)):
        rets[t] = future_ret = rewards[t] + gamma * future_ret * (~ dones[t])
    return rets

def cal_gaes(rewards, dones, v_preds, gamma, lbd):
    '''Calculate the GAE

    Parameters:
    -----------
    rewards:torch.Tensor
    [T]

    dones:torch.Tensor
    [T], bool
    
    v_preds:torch.Tensor
    [T+1]

    gamma:float

    lbd:float
    '''
    T = len(rewards);
    #[T]
    gaes = torch.zeros_like(rewards);
    #[T]
    not_dones = ~dones;
    #notes:why v_preds[1:] times not_dones, but v_preds[:-1] not?
    #adv = Q - V, (rewards + gamma*v_preds[1:]) can be understood as the estimation of Q.
    #Therefore, if the next step corresponding to reward ends, its reward is estimated to be zero.
    #And v_preds[:-1] means including rewards and next step estimates.
    #In this way, even if the next step is terminated, the estimate of the next step contained in the estimate of V is not 0, 
    # which is more conducive to the model learning to estimate 0
    delta_ts = rewards + gamma*v_preds[1:]*not_dones - v_preds[:-1];
    future_gae = torch.tensor(.0, dtype=rewards.dtype);
    for t in reversed(range(T)):
        gaes[t] = delta_ts[t] + gamma*lbd*future_gae*not_dones[t];
    return gaes;

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