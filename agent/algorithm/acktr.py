# @Time   : 2023.06.19
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.actor_critic import ActorCritic
from lib import glb_var, callback
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

    def _cal_fisher_mat(self, batch):
        if self.is_ac_shared:
            net = self.acnet;
        else:
            net = self.acnets[0];
        action_logits = self._cal_action_pd(batch['states']);
        action_pd = torch.distributions.Categorical(logits = action_logits);
        #[batch]
        log_probs = action_pd.log_prob(batch['actions']);
        #[batch]
        grads = torch.autograd.grad(log_probs.mean(), net.parameters());
        fisher_mat = torch.outer(grads, grads);
        return fisher_mat;



    def compute_fisher_matrix(self, states, actions):
        log_probs, _ = self.actor_critic(states)
        log_probs = log_probs.gather(1, actions)
        grads = torch.autograd.grad(log_probs.mean(), self.actor_critic.parameters())
        grads = torch.cat([grad.view(-1) for grad in grads])
        fisher_matrix = torch.outer(grads, grads)
        return fisher_matrix