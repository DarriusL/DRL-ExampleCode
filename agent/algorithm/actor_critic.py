# @Time   : 2023.05.30
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.reinforce import Reinforce
from agent.algorithm import alg_util
from agent.net import get_net, net_util
from lib import glb_var
import torch

logger = glb_var.get_value('log')

class SharedNetActorCritic(Reinforce):
    '''The critic network and the actor network share parts of the network'''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.action_strategy = alg_util.action_default;
        glb_var.get_value('var_reporter').add('Policy loss coefficient', self.policy_loss_var);
        if algorithm_cfg['entropy_reg_var_cfg'] is not None:
            self.entorpy_reg_var_shedule = alg_util.VarScheduler(algorithm_cfg['entropy_reg_var_cfg']);
            self.entorpy_reg_var = self.entorpy_reg_var_shedule.var_start;
        else:
            self.entorpy_reg_var_shedule = None;

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        #output1 is from actor, output2 is from critic
        out_dim = [out_dim, 1];
        self.acnet = get_net(net_cfg, in_dim, out_dim).to(glb_var.get_value('device'));
        self.optimizer = net_util.get_optimizer(optim_cfg, self.pi);
        #if None, then do not use
        self.lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, self.optimizer, max_epoch);

    def _cal_action_pd(self, state):
        '''
        Action distribution probability in the input state

        Parameters:
        ----------
        state:torch.Tensor
        '''
        #state:[..., in_dim]
        #out:[..., out_dim]
        return self.acnet(state, is_integrated = True)[0];

    def _cal_v(self, state):
        ''''''
        #state:[..., in_dim]
        #out:[..., 1] -> [...]
        return self.acnet(state, is_integrated = True)[-1].squeeze(-1);

    def _cal_mc_advs_and_v_tgts(self, batch, v_preds):
        '''Estimate Q using Monte Carlo simulations and use this to calculate advantages
        '''
        #v_preds:[batch]
        #advs and v_tgt don't need to accumulate grad
        v_preds = v_preds.detach()
        #rets:[batch]
        #Mixed trajectory, cannot use [fast]
        rets = alg_util.cal_returns(batch['rewards'], batch['dones'], self.gamma, fast = False);
        #At the same time, it is also an estimate of Q
        v_tgts = rets;
        advs = v_tgts - v_preds;
        return advs, v_tgts;

    def _cal_nstep_advs_and_v_tgts(self, batch, v_preds):
        '''Using temporal difference learning to estimate Q and then calculate the advantage'''
        #v_preds:[batch]
        #advs and v_tgt don't need to accumulate grad
        v_preds = v_preds.detach()
        with torch.no_grad():
            next_v_pred = self._cal_v(batch['states'][-1]);
        #it is the estimate of Q and also the estimate of v_tgt
        rets = alg_util.cal_nstep_returns(batch['rewards'], batch['dones'], next_v_pred, self.gamma, self.n_step_returns);
        advs = rets - v_preds;
        v_tgts = rets;
        return advs, v_tgts;

    def _cal_gae_advs_and_v_tgts(self, batch, v_preds):
        '''Calculate GAE and estimate v_tgt'''
        #v_preds:[batch]
        #advs and v_tgt don't need to accumulate grad
        v_preds = v_preds.detach()
        with torch.no_grad():
            next_v_pred = self._cal_v(batch['states'][-1]);
        #[batch+1]
        v_preds = torch.cat((v_preds, next_v_pred), dim = -1);
        #[batch]
        advs = alg_util.cal_gaes(batch['rewards'], batch['dones'], v_preds, self.gamma, self.lbd);
        #The value of Vtgt is replaced by the estimation of Q
        v_tgts = advs + v_preds[:-1]
        return advs, v_tgts
    
    def _cal_policy_loss(self, batch, rets):
        return super()._cal_loss(batch, rets);

    def _cal_value_loss(self, v_preds, v_tgts):
        return torch.nn.MSELoss()(v_preds.float(), v_tgts.float());
        

