# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from agent.net.base import Net
from agent.net import net_util

class ConvNet(Net):
    def __init__(self, net_cfg) -> None:
        super().__init__(net_cfg)

