# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.net import *
from agent.algorithm.base import Algorithm
from agent.algorithm import alg_util
from lib import glb_var
import torch

class Reinforce(Algorithm):
    '''Implementation of REINFORCE
    
    Notes:
    ------
    1.OnPolicy Algorithm
    2.Experience can only be gained using Monte Carlo simulation sampling
    3.loss:policy grad
    '''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.action_strategy = alg_util.action_default;
        if algorithm_cfg['entropy_reg_var_cfg'] is not None:
            self.entorpy_reg_var_shedule = alg_util.VarScheduler(algorithm_cfg['entropy_reg_var_cfg']);
            self.entorpy_reg_var = self.entorpy_reg_var_shedule.var_start;
        else:
            self.entorpy_reg_var_shedule = None;
    
    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        self.pi = get_net(net_cfg, in_dim, out_dim).to(glb_var.get_value('device'));
        self.optimizer = net_util.get_optimizer(optim_cfg, self.pi);
        #if None, then do not use
        self.lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, self.optimizer, max_epoch);

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
        '''Calculate returns'''
        rets = alg_util.cal_returns(batch['rewards'], batch['dones'], self.gamma);
        rets_mean = rets.mean();
        if self.rets_mean_baseline:
            rets = alg_util.rets_mean_baseline(rets);
        return rets, rets_mean.item();

    def cal_loss(self, action_batch_logits, batch):
        '''Calculate policy gradient loss for REINFORCE
        '''
        action_pd_batch = torch.distributions.Categorical(logits = action_batch_logits);
        #[T]
        log_probs = action_pd_batch.log_prob(batch['actions']);
        rets, _ = self.cal_rets(batch);
        loss = - self.policy_loss_var*(rets * log_probs).mean();

        if self.entorpy_reg_var_shedule is not None:
            #entropy regularization
            entropy_reg_loss = action_pd_batch.entropy().mean();
            loss += - self.entorpy_reg_var*entropy_reg_loss;
        return loss;

    def update(self):
        '''Update lr for REINFORCE'''
        if self.entorpy_reg_var_shedule is not None:
            self.entorpy_reg_var = self.entorpy_reg_var_shedule.step();
            glb_var.get_value('var_reporter').add('Entropy regularization coefficient', self.entorpy_reg_var);
        if self.lr_schedule is not None:
            self.lr_schedule.step();

    def train_step(self, batch):
        '''training network

        Parameters:
        -----------
        batch:dict
            Convert through batch_to_tensor before passing in
        '''
        #[T, out_dim]
        action_batch_logits = self.cal_action_pd_batch(batch);
        loss = self.cal_loss(action_batch_logits, batch);
        self.optimizer.zero_grad();
        self._check_nan(loss);
        loss.backward();
        self.optimizer.step();
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        return loss.item();
