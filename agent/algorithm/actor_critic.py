# @Time   : 2023.05.30
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.reinforce import Reinforce
from agent.algorithm import alg_util
from agent.net import get_net, net_util
from lib import glb_var, callback, util
import torch

logger = glb_var.get_value('log')

class ActorCritic(Reinforce):
    '''Actor Critic Algorithm'''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.action_strategy = alg_util.action_default;
        glb_var.get_value('var_reporter').add('Value loss coefficient', self.value_loss_var);
        
        #cal advs method
        if self.n_step_returns is not None and self.lbd is not None:
            self._cal_advs_and_v_tgts = self._cal_mc_advs_and_v_tgts;
        elif self.n_step_returns is not None and self.lbd is None:
            #use n-step
            self._cal_advs_and_v_tgts = self._cal_nstep_advs_and_v_tgts;
            glb_var.get_value('var_reporter').add('Num_step_rets', self.n_step_returns);
        elif self.n_step_returns is None and self.lbd is not None:
            #use gae
            self._cal_advs_and_v_tgts = self._cal_gae_advs_and_v_tgts;
            glb_var.get_value('var_reporter').add('GAE lambda', self.lbd);
        else:
            #simultaneously exist
            logger.error('There are two calculation methods for selecting advantages for algorithms in the configuration file');
            raise callback.CustomException('CfgError');
    

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        #output1 is from actor, output2 is from critic
        out_dim = [out_dim, 1];
        if 'name' not in net_cfg.keys():
            #not shared
            in_dim = in_dim*2;
            self.is_ac_shared = False;
        elif net_cfg['name'].lower() in ['sharedmlpnet']:
            self.is_ac_shared = True;
            logger.info('Detected as a non-shared network, the default network configuration order: actor-critic');
        else:
            logger.error('Only a non-shared network configuration');
            raise callback.CustomException('NetCfgError');
        #init
        acnet = get_net(net_cfg, in_dim, out_dim).to(glb_var.get_value('device'));
        optimizer = net_util.get_optimizer(optim_cfg, self.acnet);
        #if None, then do not use
        lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, self.optimizer, max_epoch);
        if self.is_ac_shared:
            util.set_attr(self, dict(
                acnet = acnet,
                optimizer = optimizer,
                lr_schedule = lr_schedule
            ));
        else:
            util.set_attr(self, dict(
                acnets = acnet,
                optimizers = optimizer,
                lr_schedules = lr_schedule
            ));

    def _cal_action_pd(self, state):
        '''
        Action distribution probability in the input state

        Parameters:
        ----------
        state:torch.Tensor
        '''
        #state:[..., in_dim]
        #out:[..., out_dim]
        if self.is_ac_shared:
            return self.acnet(state, is_integrated = True)[0];
        else

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
    
    def _cal_policy_loss(self, batch, advs):
        return super()._cal_loss(batch, advs);

    def _cal_value_loss(self, v_preds, v_tgts):
        loss = self.value_loss_var*torch.nn.MSELoss()(v_preds.float(), v_tgts.float());
        logger.debug(f'Critic loss: [{loss.item()}]')
        return loss;

    def train_step(self, batch):
        ''''''
        v_preds = self._cal_v(batch['states']);
        advs, v_tgts = self._cal_advs_and_v_tgts(batch, v_preds);
        policy_loss = self._cal_policy_loss(batch, advs);
        value_loss = self._cal_value_loss(v_preds, v_tgts);
        loss = policy_loss + value_loss;
        self.optimizer.zero_grad();
        self._check_nan(loss);
        loss.backward();
        #TODO: shared and not shared, different
        torch.nn.utils.clip_grad_norm_(self.acnet.parameters(), max_norm = 0.5);
        self.optimizer.step();
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        
        