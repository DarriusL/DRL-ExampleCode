# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import torch
from agent.net.base import Net
from agent.net import net_util
from lib import glb_var

class MLPNet(Net):
    '''
    Generic Multilayer Neural Networks
    
    Parameters:
    -----------
    net_cfg:dict
        Network Configuration
    '''
    def __init__(self, net_cfg, in_dim, out_dim) -> None:
        super().__init__(net_cfg)
        activation_fn = net_util.get_activation_fn(self.hid_layers_activation);
        if len(self.hid_layers) > 1:
            layers = [
                torch.nn.Linear(in_dim, self.hid_layers[0]),
                torch.nn.Dropout(glb_var.get_value('dropout_rate')),
                activation_fn] + [
                torch.nn.Linear(self.hid_layers[i], self.hid_layers[i+1]) for i in range(len(self.hid_layers) - 1)] + [
                activation_fn,
                torch.nn.Linear(self.hid_layers[-1], out_dim)    
            ];
        else:
            #len(.)==1
            layers = [
                torch.nn.Linear(in_dim, self.hid_layers[0]),
                torch.nn.Dropout(glb_var.get_value('dropout_rate')),
                activation_fn,
                torch.nn.Linear(self.hid_layers[0], out_dim)
            ]
        self.net = torch.nn.Sequential(*layers);
        #set training mode
        self.train();

    def forward(self, x):
        #x:[..., in_dim]
        #return [..., out_dim]
        return self.net(x);