# @Time   : 2023.05.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.sarsa import Sarsa
from agent.algorithm import alg_util
from agent.net import *
from agent.memory.offpolicy import PrioritizedMemory
from lib import glb_var
import torch

logger = glb_var.get_value('log')

class DQN(Sarsa):
    '''The most original Deep Q-Network algorithm
    
    Notes:
    -----
    1.OffPolicy algorithm.
    2.Data can be reused.
    3.Loss:MSELoss.
    '''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        self.is_onpolicy = False;
        self.action_strategy = alg_util.action_boltzmann;
    
    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        super().init_net(net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch);
        #use eval_net to choose action
        #use target_net to cal loss
        self.q_target_net = self.q_net;
        self.q_eval_net = self.q_net;

    def update(self):
        '''Update tau and lr for DQN'''
        self.var = self.var_schedule.step();
        logger.debug(f'{self.name} tau:[{self.var}]');
        glb_var.get_value('var_reporter').add('Tau', self.var);
        if self.lr_schedule is not None:
            self.lr_schedule.step();

    def _cal_loss(self, batch):
        '''Calculate MSELoss for DQN'''
        #[batch_size, out_dim]
        q_preds_table = self.q_net(batch['states']);
        with torch.no_grad():
            #[batch_size, out_dim]
            #choose action
            eval_next_q_preds_table = self.q_eval_net(batch['next_states']);
            #cal loss
            next_q_preds_table = self.q_target_net(batch['next_states']);
        #[batch_size]
        q_preds = q_preds_table.gather(-1, batch['actions'].unsqueeze(-1)).squeeze(-1);
        #[batch_size, 1]
        eval_action = eval_next_q_preds_table.argmax(dim = -1, keepdim = True);
        #[batch_size]
        max_next_q_pred = next_q_preds_table.gather(-1, eval_action).squeeze(-1);
        q_tar_preds = batch['rewards'] + self.gamma * max_next_q_pred * (~ batch['dones']).to(torch.float32);
        #If memory is PER, update the priority of the current batch
        mmy = glb_var.get_value('agent').memory;
        if isinstance(mmy, PrioritizedMemory):
            omegas = (q_preds.detach() - q_tar_preds.detach()).cpu().tolist();
            mmy.update_priorities(omegas);

        return torch.nn.MSELoss()(q_preds.float(), q_tar_preds.float());

    def train_step(self, batch):
        '''training network

        Parameters:
        -----------
        batch:dict
            Convert through batch_to_tensor before passing in
        '''
        loss = self.cal_loss(batch);
        self.optimizer.zero_grad();
        self._check_nan(loss);
        loss.backward();
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm = 0.5);
        self.optimizer.step();
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        logger.debug(f'{self.name} loss: [{loss.item()}]')

class TargetDQN(DQN):
    '''DQN with target network'''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        self.net_updater = net_util.NetUpdater(algorithm_cfg['net_updte_cfg']);
        glb_var.get_value('var_reporter').add('Target network update factor', self.net_updater.beta);

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        super().init_net(net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch);
        self.q_target_net = get_net(net_cfg, in_dim, out_dim).to(glb_var.get_value('device'));
        #Initialize q_target_net with q_net
        self.net_updater.net_param_copy(self.q_net, self.q_target_net);
        self.net_updater.set_net(self.q_net, self.q_target_net);
        self.q_eval_net = self.q_target_net;

    def update(self):
        '''Update tau and lr for DQN'''
        self.var = self.var_schedule.step();
        self.net_updater.update();
        logger.debug(f'{self.name} tau:[{self.var}]');
        glb_var.get_value('var_reporter').add('Tau', self.var);
        if self.lr_schedule is not None:
            self.lr_schedule.step();

class DoubleDQN(TargetDQN):
    '''Double DQN'''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg)
        
    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        super().init_net(net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch);
        self.q_eval_net = self.q_net;