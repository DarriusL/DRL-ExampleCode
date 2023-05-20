# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from lib import callback, glb_var
from agent import net
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