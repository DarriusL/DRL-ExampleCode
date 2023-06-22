# @Time   : 2023.06.19
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.actor_critic import ActorCritic
from lib import glb_var, callback
from agent.net.optimizer import KFAC
import torch

logger = glb_var.get_value('log')

class Acktr(ActorCritic):
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        self.is_onpolicy = True;
        #notes:acktr use nstep for calculate advs
        self._cal_advs_and_v_tgts = self._cal_nstep_advs_and_v_tgts;
        if self.n_step_returns is None:
            logger.error(f'Acktr use nstep to calculate advantages, but no nstep is set.');
            raise callback.CustomException('CfgError');

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch, optimizer=None):
        super().init_net(net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch, optimizer)
        if self.is_ac_shared:
            assert isinstance(self.optimizer, KFAC), f'The optimizer of [ACKTR] uses [KFAC].'
        else:
            assert isinstance(self.optimizers[0], KFAC), f'The optimizer of [ACKTR] uses [KFAC].'
            assert isinstance(self.optimizers[1], KFAC), f'The optimizer of [ACKTR] uses [KFAC].'
    
    def _suboptim_net(self, loss, net, optimizer):
        ''''''
        optimizer.acc_stats = True;
        loss.backward();
        optimizer.acc_stats = False;
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 0.5);
        self._set_shared_grads();
        optimizer.step();