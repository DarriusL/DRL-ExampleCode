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
        self.net = net_util.get_mlpnet(self.hid_layers, activation_fn, in_dim, out_dim);
        #set training mode
        self.train();

    def forward(self, x):
        #x:[..., in_dim]
        #return [..., out_dim]
        return self.net(x);

class SharedMLPNet(Net):
    def __init__(self, net_cfg, in_dim, out_dim) -> None:
        super().__init__(net_cfg);
        self.num_outnets = len(out_dim);
        assert self.num_outnets > 1
        activation_fn = net_util.get_activation_fn(self.hid_layers_activation);
        #shared body
        self.body_net = net_util.get_mlpnet(self.body_hid_layers, activation_fn, in_dim, self.body_out_dim);
        #output nets
        self.outnets =  torch.nn.ModuleList()
        for i in range(self.num_outnets):
            self.outnets.append(net_util.get_mlpnet(self.output_hid_layers, activation_fn, self.body_out_dim, out_dim[i]));
        #set training mode
        self.train();

    def forward(self, x, integrated = False):
        output = [];
        body_output = self.body_net(x);
        for outnet in range(self.outnets):
            output.append(outnet(body_output));
        if integrated:
            return output;
        else:
            return tuple(output);
