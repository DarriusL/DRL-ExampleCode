# @Time   : 2023.06.06
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm import reinforce
from agent.net import *
from agent.memory import *
from agent import alg_util
from lib import glb_var
import copy, torch

logger = glb_var.get_value('log')

class Reinforce(reinforce.Reinforce):
    '''REINFORCE with PPO
    '''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        self.clip_var_var_shedule = alg_util.VarScheduler(algorithm_cfg['clip_var_cfg']);
        self.clip_var = self.clip_var_var_shedule.var_start;
        glb_var.get_value('var_reporter').add('Clip coefficient', self.clip_var);
        self.batch_spliter = get_batch_split(self.batch_split_type);

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        super().init_net(net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch);
        #init an old pi net
        self.old_pi = copy.deepcopy(self.pi);

    def _cal_action_pd(self, state, net = None):
        '''
        Action distribution probability in the input state

        Parameters:
        ----------
        state:torch.Tensor
        '''
        #x:[..., in_dim]
        #return [..., out_dim]
        if net is not None:
            return net(state);
        return super()._cal_action_pd(state);

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

        Notes:
        ------
        Use the old policy network to select actions during training, and use the latest policy network at other times
        '''
        if is_training:
            #the logarithm of the action probability distribution
            action_logit = self._cal_action_pd(
                torch.from_numpy(state).to(torch.float32).to(glb_var.get_value('device')),
                net = self.old_pi);
            action = self.action_strategy(action_logit);
        else:
            #the logarithm of the action probability distribution
            action_logit = self._cal_action_pd(torch.from_numpy(state).to(torch.float32).to(glb_var.get_value('device')));
            action = alg_util.action_default(action_logit);
        return action.cpu().item();

    def _cal_loss(self, batch, rets):
        '''Calculate Policy loss for REINFORCE with PPO

        Parameters:
        ----------
        batch:dict

        rets:torch.tensor

        '''
        #[batch_size, out_dim]
        action_batch_logits = self._cal_action_pd(batch['states']);
        action_pd_batch = torch.distributions.Categorical(logits = action_batch_logits);
        #[batch_size]
        log_probs = action_pd_batch.log_prob(batch['actions']);
        with torch.no_grad():
            action_batch_logits_old = self._cal_action_pd(batch['states'], net = self.old_pi);
            action_pd_batch_old = torch.distributions.Categorical(logits = action_batch_logits_old);
            log_probs_old = action_pd_batch_old.log_prob(batch['actions']);
        #notes:log_probs are logits of probs
        #so probs = e^log_probs
        #probs/probs_old = e^(log_probs - log_probs)
        weights = torch.exp(log_probs - log_probs_old);
        logger.debug(f'weights: {weights}');
        loss_1 = rets * weights;
        loss_2 = rets * torch.clamp(weights, 1 - self.clip_var, 1 + self.clip_var);
        loss = - self.policy_loss_var * torch.min((loss_1, loss_2)).mean();
        logger.debug(f'cliped policy grad loss: {loss.item()}');

        if self.entorpy_reg_var_shedule is not None:
            #entropy regularization
            entropy_reg_loss = action_pd_batch.entropy().mean();
            logger.debug(f'entorpy reg loss: {entropy_reg_loss.item()}');
            loss += - self.entorpy_reg_var*entropy_reg_loss;
        return loss;

    def update(self):
        '''Update var'''
        super().update();
        self.clip_var = self.clip_var_var_shedule.step();
        glb_var.get_value('var_reporter').add('Policy gradient clipping coefficient', self.clip_var);
        #update old pi net
        net_util.net_param_copy(self.pi, self.old_pi);

    def train_step(self, batch):
        '''Train network'''
        subbatches = self.batch_spliter(batch);
        for subbatch in subbatches:
            rets = self._cal_rets(subbatch);
            loss = self._cal_loss(subbatch, rets);
            self.optimizer.zero_grad();
            self._check_nan(loss);
            loss.backward();
            self.optimizer.step();
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            logger.debug(f'Actor loss: [{loss.item()}]');

