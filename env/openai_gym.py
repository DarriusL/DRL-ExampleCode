# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from env.base import Env, _make_env
import numpy as np
import copy

class OpenaiEnv(Env):
    '''the openai environment'''
    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg);
        self.env = _make_env(env_cfg);
        self.env.reset();
        self.train_env_data = None;
        
    
    def get_state_and_action_dim(self):
        '''(state_dim, action_choice)
        '''
        return self.env.observation_space.shape[0], self.env.action_space.n;

    def get_state(self):
        return np.asarray(self.env.state, dtype=np.float32);

    def _save_train_env(self):
        '''Save the training environment for recovery'''
        self.train_env_data = {'data':copy.deepcopy(self.env), 't':self.t};
    
    def _resume_train_env(self):
        '''resume training environment'''
        self.env = self.train_env_data['data'];
        self.t = self.train_env_data['t'];
        self.train_env_data = None;

    def train(self):
        '''set train mode
        '''
        if not self.is_training:
            self.is_training = True;
            self._resume_train_env();
    
    def eval(self):
        '''set eval mode
        '''
        if self.is_training:
            self.is_training = False;
            self._save_train_env();
            self.reset();     


    def reset(self):
        ''''''
        self.t = 0;
        return self.env.reset();

    def step(self, action):
        ''''''
        self.t += 1;
        next_state, reward, done, info1, info2 = self.env.step(action);
        if self.t == self.survival_T:
            done = True;
        return next_state, reward, done, info1, info2;

    def render(self):
        ''''''
        self.env.render();


