# @Time   : 2023.06.06
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.algorithm import reinforce
from agent import alg_util
from lib import glb_var

logger = glb_var.get_value('log')

class Reinforce(reinforce.Reinforce):
    def __init__(self, algorithm_cfg) -> None:
        super().__init__(algorithm_cfg);
        self.clip_var_var_shedule = alg_util.VarScheduler(algorithm_cfg['clip_var_cfg']);
        self.clip_var = self.clip_var_var_shedule.var_start;
        glb_var.get_value('var_reporter').add('Clip coefficient', self.clip_var)