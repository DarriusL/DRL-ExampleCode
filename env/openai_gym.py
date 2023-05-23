# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from env.base import Env, _make_env
import numpy as np
import copy
from dataclasses import dataclass
from lib import glb_var

@dataclass
class Main_Body:
    env: None
    total_reward:None
    t:None

class OpenaiEnv(Env):
    '''the openai environment
    
    Notes:
    ------
    Before using the environment each time, set the mode first.
    e.g.
    >>> env = OpenaiEnv(env_cfg);
    >>> env.train();
    >>> s = env.get_state();
    ...
    >>> s, r, d, _, _ = env.step(a);
    ...
    >>> env.valid();
    ...
    The trian mode will not automatically reset the environment, but the valid mode will
    '''
    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg);
        env = _make_env(env_cfg);
        env.reset();
        self.train_env_data = None;
        total_reward = 0;
        t = 0;
        self.main_body = Main_Body(env, total_reward, t);
    
    def get_state_and_action_dim(self):
        '''(state_dim, action_choice)
        '''
        return self.main_body.env.observation_space.shape[0], self.main_body.env.action_space.n;

    def get_state(self):
        '''get the current state'''
        return np.asarray(self.main_body.env.state, dtype=np.float32);

    def get_total_reward(self):
        '''Get the total rewards of the current trajectory so far'''
        return self.main_body.total_reward;

    def is_terminated(self):
        '''Is the current environment terminated'''
        return True if self.main_body.env.steps_beyond_terminated is not None else False;

    def _save_train_env(self):
        '''Save the training environment for recovery'''
        self.train_env_data = copy.deepcopy(self.main_body);
    
    def _resume_train_env(self):
        '''resume training environment'''
        self.main_body = self.train_env_data;
        self.train_env_data = None;

    def train(self):
        '''set train mode
        '''
        if (not self.is_training):
            self.is_training = True;
            if glb_var.get_value('mode') == 'train':
                self._resume_train_env();
    
    def valid(self):
        '''set valid mode
        '''
        if self.is_training :
            self.is_training = False;
            if glb_var.get_value('mode') == 'train':
                self._save_train_env();
        self.reset();     


    def reset(self):
        ''''''
        self.main_body.total_reward = 0;
        self.main_body.t = 0;
        return self.main_body.env.reset();

    def step(self, action):
        ''''''
        self.main_body.t += 1;
        next_state, reward, done, info1, info2 = self.main_body.env.step(action);
        self.main_body.total_reward += reward;
        if self.main_body.t == self.survival_T:
            done = True;
        return next_state, reward, done, info1, info2;

    def render(self):
        ''''''
        self.main_body.env.render();


