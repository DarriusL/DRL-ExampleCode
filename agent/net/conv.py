# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from agent.net.base import Net
from agent.net import net_util

class ConvNet(Net):
    '''

    Parameters:
    ----------
    net_cfg: dict
        configuration of the net

    in_dim: list 
        dimension of the input:[c, h, w]
    
    out_dim: list
     dimension of the output

    
    CfgExample:
    -----------

    "net_cfg":{
        "name":"ConvNet",
        "conv_hid_layers":[
            [32, 8, 4, 0, 1],
            [32, 4, 2, 0, 1]
        ],
        "fc_hid_layers":[512],
        "hid_layers_activation":"relu",
        "out_layer_activation":"tanh",
        "normalize":true,
        ""batch_norm":true
    }
    '''
    def __init__(self, net_cfg, in_dim, out_dim) -> None:
        super().__init__(net_cfg);
        self.in_dim = in_dim;
        self.body = net_util.get_conv2d_net(
            in_channel = in_dim[0], 
            conv_hid_layers = self.conv_hid_layers, 
            activation_fn = net_util.get_activation_fn(self.hid_layers_activation), 
            batch_norm = self.batch_norm);
        conv_out_dim = self._get_conv_out_dim();
        if len(out_dim) == 1:
            self.tail = net_util.get_mlp_net(
                hid_layers = self.fc_hid_layers,
                activation_fn = net_util.get_activation_fn(self.out_layer_activation),
                in_dim = conv_out_dim,
                out_dim = out_dim[0]);
        else:
            tails = [];
            for out_d in out_dim:
                tails.append(
                    net_util.get_mlp_net(
                    hid_layers = self.fc_hid_layers,
                    activation_fn = net_util.get_activation_fn(self.out_layer_activation),
                    in_dim = conv_out_dim,
                    out_dim = out_d)
                )
            self.tails = torch.nn.ModuleList(tails);
        self.train();
        
    def _get_conv_out_dim(self):
        '''Get flatted output size of  conv net
        '''
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.body(x)
            return x.numel()
    
    def forward(self, x):
        '''
        x:[b, c, h, w]
        '''
        if self.normalize:
            x = x/255;
        y_conv = self.body(x).reshape(x.shape[0], -1);
        if hasattr(self, 'tails'):
            y = [];
            for tail in self.tails:
                y.append(tail(y_conv));
            return y;
        else:
            return self.tail(y_conv);
        

