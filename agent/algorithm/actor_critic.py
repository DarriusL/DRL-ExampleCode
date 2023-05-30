# @Time   : 2023.05.30
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm.reinforce import Reinforce

class ActorCritic(Reinforce):
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg)