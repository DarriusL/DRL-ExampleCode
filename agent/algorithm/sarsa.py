# @Time   : 2023.05.19
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.net import *
from agent.algorithm.base import Algorithm
from agent.algorithm import alg_util
from lib import glb_var
import torch

class Sarsa(Algorithm):
    '''State-Action-Reward-State-Action algorithm
    
    Notes:
    ------
    1.OnPolicy algorithm.
    2.The data sampling method can be selected between Monte Carlo sampling and nstep.
    3.Loss:MSEloss
    4.Action Policy: epsilon greedy
    '''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.action_strategy = alg_util.action_epsilon_greedy;
        self.var_schedule = alg_util.VarScheduler(algorithm_cfg['var_schedule_cfg']);
        self.var = self.var_schedule.var_start;

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        self.q_net = get_net(net_cfg, in_dim, out_dim, device = glb_var.get_value('device'));
        self.optimizer = net_util.get_optimizer(optim_cfg, self.q_net);
        glb_var.get_value('var_reporter').add('lr', self.optimizer.param_groups[0]["lr"])
        #if None, then do not use
        self.lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, self.optimizer, max_epoch);

    def update(self):
        '''Update epsilon and lr for SARSA'''
        self.var = self.var_schedule.step();
        glb_var.get_value('var_reporter').add('Epsilon', self.var);
        glb_var.get_value('logger').debug(f'{self.name} epsilon:[{self.var}]');
        if self.lr_schedule is not None:
            self.lr_schedule.step();
            glb_var.get_value('var_reporter').add('lr', self.optimizer.param_groups[0]["lr"])
        
    def _cal_action_pd(self, state):
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
        action_logit = self.q_net(torch.from_numpy(state).to(torch.float32).to(glb_var.get_value('device')));
        if is_training:
            action = self.action_strategy(action_logit, self.var);
        else:
            action = alg_util.action_default(action_logit);
        return action.cpu().item();

    def _cal_loss(self, batch):
        '''Calculate MSELoss for SARSA'''
        #[T, out_dim]
        q_preds_table = self.q_net(batch['states']);
        #[T, out_dim]
        with torch.no_grad():
            next_q_preds_table = self.q_net(batch['next_states']);
        #[T]
        q_preds = q_preds_table.gather(-1, batch['actions'].unsqueeze(-1)).squeeze(-1);
        #[T]
        next_q_preds = next_q_preds_table.gather(-1, batch['next_actions'].unsqueeze(-1)).squeeze(-1);
        q_tar_preds = batch['rewards'] + self.gamma * next_q_preds * (~ batch['dones']).to(torch.float32);
        return torch.nn.MSELoss()(q_preds.float(), q_tar_preds.float());

    def train_step(self, batch):
        '''training network

        Parameters:
        -----------
        batch:dict
            Convert through batch_to_tensor before passing in
        '''
        batch['next_actions'] = torch.zeros_like(batch['actions'])
        batch['next_actions'][:-1] = batch['actions'][1:]
        loss = self._cal_loss(batch);
        self.optimizer.zero_grad();
        self._check_nan(loss);
        loss.backward();
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm = 0.5);
        self.optimizer.step();
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        glb_var.get_value('logger').debug(f'SARSA loss: [{loss.item()}]')
