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
    '''Actor Critic Algorithm
    
    Notes:
    ------
    1.OnPolicy algorithm

    '''
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        #label for onpolicy algorithm
        self.is_onpolicy = True;
        self.action_strategy = alg_util.action_default;
        glb_var.get_value('var_reporter').add('Value loss coefficient', self.value_loss_var);
        
        #cal advs method
        if self.n_step_returns is None and self.lbd is None:
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
    

    def init_net(self, net_cfg, optim_cfg, lr_schedule_cfg, in_dim, out_dim, max_epoch, optimizer = None):
        '''Initialize the network and initialize optimizer and learning rate scheduler
        '''
        #output1 is from actor, output2 is from critic
        out_dim = [out_dim, 1];
        if 'name' not in net_cfg.keys():
            #not shared
            in_dim = [in_dim]*2;
            self.is_ac_shared = False;
        elif net_cfg['name'].lower() in ['sharedmlpnet']:
            self.is_ac_shared = True;
            logger.info('Detected as a shared network, the default network configuration order: actor-critic');
        else:
            logger.error('Only a non-shared network configuration');
            raise callback.CustomException('NetCfgError');
        #init
        acnet = get_net(net_cfg, in_dim, out_dim, device = glb_var.get_value('device'));
        if optimizer is None:
            #if is not none, then is a3c train sys
            optimizer = net_util.get_optimizer(optim_cfg, acnet);
        #if None, then do not use
        lr_schedule = net_util.get_lr_schedule(lr_schedule_cfg, optimizer, max_epoch);
        if self.is_ac_shared:
            util.set_attr(self, dict(
                acnet = acnet,
                optim_net = acnet,
                optimizer = optimizer,
                lr_schedule = lr_schedule
            ));
            glb_var.get_value('var_reporter').add('lr', self.optimizer.param_groups[0]["lr"]);
        else:
            util.set_attr(self, dict(
                acnets = acnet,
                optim_nets = acnet,
                optimizers = optimizer,
                lr_schedules = lr_schedule
            ));
            glb_var.get_value('var_reporter').add('actor-lr', self.optimizers[0].param_groups[0]["lr"]);
            glb_var.get_value('var_reporter').add('critic-lr', self.optimizers[-1].param_groups[0]["lr"]);
    
    def get_optimizer(self):
        ''''''
        if self.is_ac_shared:
            return self.optimizer;
        else:
            return self.optimizers;
    
    def share_memory(self):
        '''Share Net memory in A3C algorithm'''
        if self.is_ac_shared:
            self.acnet.share_memory();
        else:
            self.acnets[0].share_memory();
            self.acnets[1].share_memory();
    
    def set_shared_net(self, alg):
        '''Set shared net in A3C algorithm'''
        if self.is_ac_shared:
            self.shared_net = alg.acnet;
        else:
            self.shared_nets = alg.acnets;
    
    def load_sharednet(self):
        '''Load shared net is A3C algorithm'''
        if self.is_ac_shared:
            net_util.net_param_copy(self.shared_net, self.acnet);
        else:
            net_util.net_param_copy(self.shared_nets[0], self.acnets[0]);
            net_util.net_param_copy(self.shared_nets[1], self.acnets[1]);
    
    def _set_shared_grads(self):
        ''''''
        def __set_shared_param(net, shared_net):
            for param, shared_param in zip(net.parameters(), shared_net.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad
                else:
                    shared_param.grad += param.grad
        if self.is_asyn:
            if self.is_ac_shared:
                __set_shared_param(self.acnet, self.shared_net);
            else:
                for net, shared_net in zip(self.acnet, self.shared_nets):
                    __set_shared_param(net, shared_net);
        else:
            pass;

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
        else:
            return self.acnets[0](state);

    def _cal_v(self, state):
        ''''''
        #state:[..., in_dim]
        #out:[..., 1] -> [...]
        if self.is_ac_shared:
            return self.acnet(state, is_integrated = True)[-1].squeeze(-1);
        else:
            return self.acnets[-1](state).squeeze(-1);

    def _cal_mc_advs_and_v_tgts(self, batch, v_preds):
        '''Estimate Q using Monte Carlo simulations and use this to calculate advantages
        '''
        #v_preds:[batch]
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
        with torch.no_grad():
            #is a value
            next_v_pred = self._cal_v(batch['states'][-1]);
        #it is the estimate of Q and also the estimate of v_tgt
        rets = alg_util.cal_nstep_returns(batch['rewards'], batch['dones'], next_v_pred, self.gamma, self.n_step_returns);
        advs = rets - v_preds;
        v_tgts = rets;
        return advs, v_tgts;

    def _cal_gae_advs_and_v_tgts(self, batch, v_preds):
        '''Calculate GAE and estimate v_tgt'''
        #v_preds:[batch]
        with torch.no_grad():
            #[1]
            next_v_pred = self._cal_v(batch['states'][-1].unsqueeze(0));
        #[batch+1]
        v_preds = torch.cat((v_preds, next_v_pred), dim = -1);
        #[batch]
        advs = alg_util.cal_gaes(batch['rewards'], batch['dones'], v_preds, self.gamma, self.lbd);
        #The value of Vtgt is replaced by the estimation of Q
        v_tgts = advs + v_preds[:-1]
        return advs, v_tgts
    
    def _cal_policy_loss(self, batch, advs):
        ''''''
        return super()._cal_loss(batch, advs);

    def _cal_value_loss(self, v_preds, v_tgts):
        ''''''
        loss = self.value_loss_var*torch.nn.MSELoss()(v_preds.float(), v_tgts.float());
        logger.debug(f'Critic loss: [{loss.item()}]')
        return loss;

    def update(self):
        '''Update'''
        if self.entorpy_reg_var_shedule is not None:
            self.entorpy_reg_var = self.entorpy_reg_var_shedule.step();
            glb_var.get_value('var_reporter').add('Entropy regularization coefficient', self.entorpy_reg_var);
        if self.is_ac_shared and self.lr_schedule is not None:
            self.lr_schedule.step();
            glb_var.get_value('var_reporter').add('lr', self.optimizer.param_groups[0]["lr"]);
        elif not self.is_ac_shared and self.lr_schedules is not None:
            for schduler in self.lr_schedules:
                schduler.step();
            glb_var.get_value('var_reporter').add('actor-lr', self.optimizers[0].param_groups[0]["lr"]);
            glb_var.get_value('var_reporter').add('critic-lr', self.optimizers[-1].param_groups[0]["lr"]);

    
    def train_step(self, batch):
        ''''''
        with torch.no_grad():
            v_preds = self._cal_v(batch['states']);
            advs, v_tgts = self._cal_advs_and_v_tgts(batch, v_preds);
        self._train_main(batch, advs, v_tgts);

    def _train_main(self, batch, advs, v_tgts):
        ''''''
        v_preds = self._cal_v(batch['states']);
        policy_loss = self._cal_policy_loss(batch, advs);
        self._check_nan(policy_loss);
        value_loss = self._cal_value_loss(v_preds, v_tgts);
        self._check_nan(value_loss);
        if self.is_ac_shared:
            loss = policy_loss + value_loss;
            self.optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(self.acnet.parameters(), max_norm = 0.5);
            self._set_shared_grads();
            self.optimizer.step();
        else:
            for net, optimzer, loss in zip(self.acnets, self.optimizers, [policy_loss, value_loss]):
                optimzer.zero_grad();
                loss.backward();
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 0.5);
                self._set_shared_grads();
                optimzer.step();
            loss = policy_loss + value_loss;
        logger.debug(f'ActorCritic Total loss:{loss.item()}');
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        
        