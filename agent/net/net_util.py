# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch, math, kfac
from lib import callback, glb_var, util
from agent.net import *

logger = glb_var.get_value('log');

def get_optimizer(optim_cfg, net):
    '''Get Network Parameter Optimizer

    Parameters:
    -----------
    optim_cfg:dict
        configuration of optimizer

    net:torch.Module
        Networks that need to optimize parameters
    '''
    if 'name' not in optim_cfg.keys():
        #optim cfgs for nets
        optims = [];
        nets, optim_cfgs = net, optim_cfg;
        assert len(optim_cfgs) == len(nets), 'The length of optim_cfgs and nets should be the same';
        
        for i, cfg in enumerate(optim_cfgs.values()):
            optims.append(get_optimizer(cfg, nets[i]))
        return optims;

    if optim_cfg['name'].lower() == 'adam':
        return torch.optim.Adam(net.parameters(), lr = optim_cfg['lr'], betas = optim_cfg['betas'], weight_decay = optim_cfg['weight_decay']);
    elif optim_cfg['name'].lower() == 'sharedadam':
        return SharedAdam(net.parameters(), lr = optim_cfg['lr'], betas = optim_cfg['betas'], weight_decay = optim_cfg['weight_decay']);
    elif optim_cfg['name'].lower() == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr = optim_cfg['lr'], betas = optim_cfg['betas'], weight_decay = optim_cfg['weight_decay']);
    elif optim_cfg['name'].lower() == 'rmsprop':
        return torch.optim.RMSprop(net.parameters(), lr = optim_cfg['lr'], alpha = optim_cfg['alpha'], weight_decay = optim_cfg['weight_decay']);
    elif optim_cfg['name'].lower() == 'kfac':
        return kfac.KfacOptimizer(net.parameters(), lr = optim_cfg['lr'], weight_decay = optim_cfg['weight_decay']);
    else:
        logger.warning(f"Unrecognized optimizer[{optim_cfg['name']}], set default Adam optimizer");
        return torch.optim.Adam(net.parameters(), lr = optim_cfg['lr']);

def get_lr_schedule(lr_schedule_cfg, optimizer, max_epoch):
    '''Get the learning rate scheduler
    
    Parameters:
    -----------
    lr_schedule_cfg:dict
        Configuration of the learning rate scheduler
    
    optimzer:torch.optim
        An optimizer that requires learning rate scheduling
    
    max_epoch:int
    '''
    if lr_schedule_cfg is None:
        return None;

    if 'name' not in lr_schedule_cfg.keys():
        #cfgs for multiple optimizers
        schedulers = [];
        lr_schedule_cfgs, optimizers = lr_schedule_cfg, optimizer;
        for i, cfg in enumerate(lr_schedule_cfgs.values()):
            schedulers.append(get_lr_schedule(cfg, optimizers[i], max_epoch));
        return schedulers;

    if lr_schedule_cfg['name'].lower() == 'onecyclelr':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer, 
            max_lr = lr_schedule_cfg['lr_max'],
            total_steps = max_epoch);
    elif lr_schedule_cfg['name'].lower() == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer = optimizer,
            step_size = lr_schedule_cfg['step_size'],
            gamma = lr_schedule_cfg['gamma']
        )
    elif lr_schedule_cfg['name'].lower() == 'exponentiallr':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = lr_schedule_cfg['gamma']
        )
    else:
        logger.error(f'Type of schedule [{lr_schedule_cfg["name"]} is not supported.]');
        raise callback.CustomException('LrScheduleError')

def get_activation_fn(name = 'selu'):
    '''
    Get the activation function

    Parameters:
    ----------
    name: str
        the name of the activation function
        default: 'selu'
    '''
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax', 'selu'];
    if name.lower() == 'sigmoid':
        return torch.nn.Sigmoid();
    elif name.lower() == 'tanh':
        return torch.nn.Tanh();
    elif name.lower() == 'relu':
        return torch.nn.ReLU();
    elif name.lower() == 'leaky_relu':
        return torch.nn.LeakyReLU();
    elif name.lower() == 'elu':
        return torch.nn.ELU();
    elif name.lower() == 'softmax':
        return torch.nn.Softmax();
    elif name.lower() == 'selu':
        return torch.nn.SELU();
    else:
        logger.error(f'Activation function [{name.lower()}] does not support automatic acquisition at the moment,'
                                        f'please replace or add the code yourself.\nSupport list:{activations}');
        raise callback.CustomException('ActivationCfgNameError');

def get_mlpnet(hid_layers, activation_fn, in_dim, out_dim):
    ''''''
    if len(hid_layers) > 1:
        layers = [
            torch.nn.Linear(in_dim, hid_layers[0]),
            torch.nn.Dropout(glb_var.get_value('dropout_rate')),
            activation_fn] + [
            torch.nn.Linear(hid_layers[i], hid_layers[i+1]) for i in range(len(hid_layers) - 1)] + [
            activation_fn,
            torch.nn.Linear(hid_layers[-1], out_dim)    
        ];
    else:
        #len(.)==1
        layers = [
            torch.nn.Linear(in_dim, hid_layers[0]),
            torch.nn.Dropout(glb_var.get_value('dropout_rate')),
            activation_fn,
            torch.nn.Linear(hid_layers[0], out_dim)
        ]
    return torch.nn.Sequential(*layers);

def net_param_copy(src, tgt):
    '''Copy network parameters from src to tgt'''
    tgt.load_state_dict(src.state_dict());

class NetUpdater():
    '''for updating the network'''
    def __init__(self, net_update_cfg) -> None:
        util.set_attr(self, net_update_cfg, except_type = dict);
        self.epoch = 0;
        #generate net update policy
        if self.name.lower() == 'replace':
            self.updater = net_param_copy;
        elif self.name.lower() == 'polyak':
            self.updater = self.net_param_polyak_update;
        else:
            logger.error(f'Unsupported type {self.name}, '
                                              'implement it yourself or replace it with [replace, polyak]');

    def set_net(self, src, tgt):
        ''''''
        self.src_net = src;
        self.tgt_net = tgt;

    def net_param_polyak_update(self, src, tgt):
        '''Polyak updata policy
        
        Parameters:
        ----------
        beta:coefficient of network update
            tgt = beta * tgt + (1- beta)*src
        '''
        for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
            tgt_param.data.copy_(self.beta*tgt_param.data + (1 - self.beta)*src_param.data);

    def update(self):
        self.epoch += 1;
        if self.epoch % self.update_step == 0:
            self.updater(self.src_net, self.tgt_net);
            logger.debug('Net update.');

class SharedAdam(torch.optim.Adam):
    '''
    Implements Adam algorithm with shared states.
    Adapted from [1].

    Refrence:
    ---------
    [1]https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py
    [2]https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py
    [3]https://zhuanlan.zhihu.com/p/346205754
    '''
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = torch.add(grad, p.data, alpha = group['weight_decay']);

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value = -step_size)

        return loss