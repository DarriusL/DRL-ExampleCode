# @Time   : 2023.05.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.sarsa import Sarsa
from agent.algorithm import alg_util
from lib import glb_var
import torch

class ClassicDQN(Sarsa):
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
        #Mark: Used to update the learning rate after training, 
        #because it needs to be collected in a certain period of epoch and cannot be trained
        self.is_train_point = False;
        self.action_strategy = alg_util.action_boltzmann;

    def update(self):
        '''Update tau and lr for DQN'''
        self.var = self.var_schedule.step();
        if (self.lr_schedule is not None) and self.is_train_point:
            self.lr_schedule.step();
            self.is_train_point = True;
        glb_var.get_value('logger').debug(f'{self.name} tau:[{self.var}]');

    def cal_loss(self, batch):
        '''Calculate MSELoss for DQN'''
        #[batch_size, out_dim]
        q_preds_table = self.q_net(batch['states']);
        with torch.no_grad():
            #[batch_size, out_dim]
            next_q_preds_table = self.q_net(batch['next_states']);
        #[batch_size]
        q_preds = q_preds_table.gather(-1, batch['actions'].unsqueeze(-1)).squeeze(-1);
        #[batch_size]
        max_next_q_pred, _ = next_q_preds_table.max(dim = -1);
        q_tar_preds = batch['rewards'] + self.gamma * max_next_q_pred * (~ batch['dones']).to(torch.float32);
        return torch.nn.MSELoss()(q_preds.float(), q_tar_preds.float());

    def train_step(self, batch):
        '''training network

        Parameters:
        -----------
        batch:dict
            Convert through batch_to_tensor before passing in
        '''
        self.is_train_point = True;
        loss = self.cal_loss(batch);
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        self.optimizer.zero_grad();
        self._check_nan(loss);
        loss.backward();
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm = 0.5);
        self.optimizer.step();
        if self.lr_schedule is not None:
            self.lr_schedule.step();
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        return loss.item();