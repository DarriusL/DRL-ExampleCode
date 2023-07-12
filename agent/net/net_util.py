# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from lib import callback, glb_var, util
from agent.net import *
from agent.net.optimizer import *

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
        return KFAC(net, lr = optim_cfg['lr'], weight_decay = optim_cfg['weight_decay']);
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

def get_mlp_net(hid_layers, activation_fn, in_dim, out_dim):
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

def get_conv2d_net(in_channel, conv_hid_layers, activation_fn, batch_norm = False):
    '''
    Parameters:
    -----------
    channel_in:int
        the channel of th input imag data

    conv_hid_layers:list
        parameters of the conv2d input: [out_channel, kernel, stride, padding, dialation]
    
    activation_fn: 
        activation function
    '''
    conv_layers = []; 
    for i, layer in enumerate(conv_hid_layers):
        conv_layers.append(torch.nn.Conv2d(in_channel, *layer));
        conv_layers.append(activation_fn);
        if batch_norm and i != len(conv_hid_layers) - 1:
            conv_layers.append(torch.nn.BatchNorm2d(layer[0]));
        in_channel = layer[0];
    return torch.nn.Sequential(*conv_layers);


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