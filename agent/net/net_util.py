# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from lib import callback, glb_var, util
from agent.net import *

def get_optimizer(optim_cfg, net):
    '''
    Get Network Parameter Optimizer
    '''
    if optim_cfg['name'].lower() == 'adam':
        return torch.optim.Adam(net.parameters(), lr = optim_cfg['lr'], betas = optim_cfg['betas'], weight_decay = optim_cfg['weight_decay']);
    elif optim_cfg['name'].lower() == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr = optim_cfg['lr'], betas = optim_cfg['betas'], weight_decay = optim_cfg['weight_decay']);
    elif optim_cfg['name'].lower() == 'rmsprop':
        return torch.optim.RMSprop(net.parameters(), lr = optim_cfg['lr'], alpha = optim_cfg['alpha'], weight_decay = optim_cfg['weight_decay']);
    else:
        glb_var.get_value('logger').warning(f"Unrecognized optimizer[{optim_cfg['name']}], set default Adam optimizer");
        return torch.optim.Adam(net.parameters(), lr = optim_cfg['lr']);

def get_lr_schedule(lr_schedule_cfg, optimizer, max_epoch):
    '''Get the learning rate scheduler'''
    if lr_schedule_cfg is None:
        return None;
    elif lr_schedule_cfg['name'].lower() == 'onecyclelr':
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
        glb_var.get_value('logger').error(f'Type of schedule [{lr_schedule_cfg["name"]} is not supported.]');
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
        glb_var.get_value('logger').error(f'Activation function [{name.lower()}] does not support automatic acquisition at the moment,'
                                        f'please replace or add the code yourself.\nSupport list:{activations}');
        raise callback.CustomException('ActivationCfgNameError');

class NetUpdater():
    '''for updating the network'''
    def __init__(self, net_update_cfg) -> None:
        util.set_attr(self, net_update_cfg, except_type = dict);
        self.epoch = 0;
        #generate net update policy
        if self.name.lower() == 'replace':
            self.updater = self.net_param_copy;
        elif self.name.lower() == 'polyak':
            self.updater = self.net_param_polyak_update;
        else:
            glb_var.get_value('logger').error(f'Unsupported type {self.name}, '
                                              'implement it yourself or replace it with [replace, polyak]');

    def set_net(self, src, tgt):
        ''''''
        self.src_net = src;
        self.tgt_net = tgt;

    def net_param_copy(self, src, tgt):
        '''Copy network parameters from src to tgt'''
        tgt.load_state_dict(src.state_dict());

    def net_param_polyak_update(self, src, tgt):
        '''Polyak updata policy
        
        Parameters:
        ----------
        beta:coefficient of network update
            tgt = beta * tgt + (1- beta)*src
        '''
        for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
            tgt.data.copy_(self.beta*tgt_param.data + (1 - self.beta)*src_param.data);

    def update(self):
        self.epoch += 1;
        if self.epoch % self.update_step == 0:
            self.updater(self.src_net, self.tgt_net);
            glb_var.get_value('logger').debug('Net update.')