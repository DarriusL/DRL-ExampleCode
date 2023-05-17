# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from agent import algorithm
from agent.algorithm import *
from lib import glb_var, callback

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

def rets_mean_baseline(rets):
    '''Add baselines to returns
    '''
    return rets - rets.mean();