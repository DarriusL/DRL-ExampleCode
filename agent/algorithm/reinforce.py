# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.net import net_util
from agent.algorithm.base import Algorithm
from agent.algorithm import alg_util
from lib import glb_var
import torch
import numpy as np

class Reinforce(Algorithm):
    '''Implementation of REINFORCE
    '''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #reset algorithm
        self.reset();
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.to_train = False;

    def _init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        self.pi = net_util.get_net(net_cfg, in_dim, out_dim);
        self.optimizer = net_util.get_optimizer(optim_cfg, self.pi);
        #if None, then do not use
        self.lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, self.optimizer);

    def batch_to_tensor(self, batch):
        '''Convert a batch to a format for torch training'''
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
        return self.pi(state);

    def cal_action_pd_batch(self, batch):
        '''Calculate the logarithm value of the action probability of a batch

        Parameters:
        -----------
        batch:dict
            The value in it has been converted to tensor
        '''
        states_batch = batch['states'];
        return self.cal_action_pd(states_batch);

    @torch.no_grad()
    def act(self, state):
        '''take action on input state

        Parameters:
        -----------
        state:numpy.ndarray

        Returns:
        --------
        action
        '''
        #the logarithm of the action probability distribution
        action_logtis = self.pi(torch.from_numpy(state).to(torch.float32));
        #action probability distribution
        action_pd = torch.distributions.Categorical(logits = action_logtis);
        action = action_pd.sample();
        return action.item();


    def cal_loss(self, action_batch_logits, batch):
        '''Calculate policy gradient loss for REINFORCE
        '''
        action_pd_batch = torch.distributions.Categorical(logits = action_batch_logits);
        #[T]
        log_probs = action_pd_batch.log_prob(batch['actions']);
        rets = alg_util.cal_returns(batch['rewards'], batch['dones'], self.gamma);
        #Add baselines to returns
        if self.rets_mean_baseline:
            rets = alg_util.rets_mean_baseline(rets);
        loss = - (rets * log_probs).mean();
        return loss
    
    def train(self, batch):
        if self.to_train:
            batch = self.batch_to_tensor(batch);
            #[T, out_dim]
            action_batch_logits = self.cal_action_pd_batch(batch);
            loss = self.cal_loss(action_batch_logits, batch);
            if self.lr_schedule is not None:
                self.lr_schedule.step();
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            self.optimizer.zero_grad();
            self._check_nan(loss);
            loss.backward();
            self.optimizer.step();
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            return loss.item();
        else:
            return None
