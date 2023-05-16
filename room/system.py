# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory import *
from agent.algorithm.alg_util import get_alg
from collections import namedtuple
from lib import glb_var
from env import *

class System():
    def __init__(self, cfg) -> None:
        self.cfg = cfg;
        self.loss = [];
        self.rets_mean = [];
        self.logger = glb_var.get_value('logger');
        memory = get_memory(cfg['agent_cfg']['memory_cfg']);
        algorithm = get_alg(cfg['agent_cfg']['algorithm_cfg'])
        self.agent = namedtuple('Agent', ['memory', 'algorithm'])(memory, algorithm);
        self.env = make_env(cfg['env']);
        in_dim = self.env.observation_space.shape[0];
        out_dim = self.env.action_space.n;
        #Initialize the agent's network
        self.agent.algorithm.init_net(
            cfg['agent_cfg']['net_cfg'],
            cfg['agent_cfg']['optimizer_cfg'],
            cfg['agent_cfg']['lr_schedule_cfg'],
            in_dim,
            out_dim,
            cfg['max_epoch']
        );
    
    def _check_train_point(self):
        if len(self.agent.memory.states) == self.cfg['train_exp_size']:
            self.logger.debug(f'The experience length [{len(self.agent.memory.states)}]reaches the training experience length, '
                              'and the training will start');
            return True;
        else:
            self.logger.debug(f'Current experience length [{len(self.agent.memory.states)}]' 
                              f'does not meet the training requirements [{self.cfg["train_exp_size"]}].'
                              'The agent will continue to gain experience');
            return False;

    def train(self):
        for epoch in range(self.cfg['agent_cfg']['max_epoch']):
            state = self.env.reset();
            for _ in range(self.cfg['agent_cfg']['T']):
                action = self.agent.algorithm.act(state);
                next_state, reward, done, _ = self.env.step(action);
                self.agent.memory.update(state, action, reward, next_state, done);
                if done:
                    break;
            
            if self._check_train_point():
                #start to train
                batch = self.agent.memory.sample();
                loss, rets_mean = self.agent.algorithm.train_epoch(batch);
                self.loss.append(loss);
                self.rets_mean.append(rets_mean);
                total_rewards = sum(batch['rewards']);
                solved = total_rewards > self.cfg['env']['solved_total_reward'];
                self.logger(f'[train - {self.agent.algorithm.name} - {self.agent.memory.name} - {self.env.name}]/n'
                            f'Epoch: [{epoch}/{self.cfg["agent_cfg"]["max_epoch"]}] - train loss: {loss:.8f} - '
                            f'lr: {self.agent.algorithm.optimizer.param_groups[0]["lr"]}\n'
                            f'Mean Returns: [{rets_mean}] - Total Rewards: [{total_rewards}] - solved: {solved}');


