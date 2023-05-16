# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.net.net_util import get_net
from agent.algorithm.base import Algorithm

class Reinforce(Algorithm):
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg)
