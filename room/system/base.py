# @Time   : 2023.05.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import glb_var, json_util, util
from dataclasses import dataclass
from agent.memory import *
import os

@dataclass
class Agent:
    memory:None
    algorithm:None

class System():
    '''Abstract Memory class to define the API methods'''
    def __init__(self, cfg, algorithm, env) -> None:
        self.logger = glb_var.get_value('logger');
        self.cfg = cfg;
        self.env = env;
        if glb_var.get_value('mode') == 'test':
            self.save_path, _ = os.path.split(self.cfg['model_path']);
        elif glb_var.get_value('mode') == 'train':
            self.save_path = glb_var.get_value('save_dir') + f'/{algorithm.name.lower()}/{self.env.name.lower()}/' \
                            f'[{util.get_date("_")}]_[{util.get_time("_")}]';
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path);
            self.cfg['model_path'] = self.save_path + '/alg.model';
            json_util.jsonsave(self.cfg, self.save_path + '/config.json'); 
        memory = get_memory(cfg['agent_cfg']['memory_cfg']);
        self.agent = Agent(memory, algorithm);
        util.set_attr(self.agent, cfg['agent_cfg'], except_type = dict)
        self.env.reset();

    def _check_mode(self):
        '''Check whether it is dev mode'''
        if glb_var.get_value('dev'):
            input("\n>>>press any key to continue<<<");

    def _explore(self):
        '''The agent explores the environment to gain experience'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train(self):
        '''train agent'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def test(self):
        '''train agent'''
        glb_var.get_value("logger").error('Method needs to be called after being implemented');
        raise NotImplementedError;