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
    state: None
    total_reward:None
    t:None
    is_terminated:None

image_envs = ['pong'];
#TODO:a new verison
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
    The trian mode will not automatically reset the environment when switch, but the valid mode will
    '''
    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg);
        env = _make_env(env_cfg);
        env.reset();
        self.train_env_data = None;
        total_reward = 0;
        t = 0;
        self.main_body = Main_Body(env, total_reward, t, False);
    
    def _transpose(self, state):
        return state.transpose((2, 0, 1));
    
    def get_state_and_action_dim(self):
        '''(state_dim, action_choice)
        '''
        if self.name.lower() in image_envs:
            state_dim = self._transpose(self.main_body.env.reset()[0]).shape;
        else:
            state_dim = self.main_body.env.observation_space.shape[0];
        return state_dim, self.main_body.env.action_space.n;

    def get_state(self):
        '''get the current state'''
        return np.asarray(self.main_body.state, dtype=np.float32);

    def get_total_reward(self):
        '''Get the total rewards of the current trajectory so far'''
        return self.main_body.total_reward;

    def is_terminated(self):
        '''Is the current environment terminated'''
        return self.main_body.is_terminated;

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
        if not self.is_training:
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

    def set_seed(self, seed):
        '''Set the seed of the env.'''
        np.random.seed(seed);

    def reset(self):
        '''Reset the env'''
        self.main_body.total_reward = 0;
        self.main_body.t = 0;
        self.main_body.is_terminated = False;
        state, _ = self.main_body.env.reset();
        if self.name.lower() in image_envs:
            state = self._transpose(state);
        self.main_body.state = state;
        return state;

    def step(self, action):
        '''Change the env through the action'''
        if self.main_body.is_terminated:
            raise RuntimeError
        self.main_body.t += 1;
        next_state, reward, done, info1, info2 = self.main_body.env.step(action);
        if self.name.lower() in image_envs:
            next_state = self._transpose(next_state);
        self.main_body.state = next_state;
        self.main_body.total_reward += reward;
        if self.main_body.t == self.survival_T:
            done = True;
        self.main_body.is_terminated = done;
        return next_state, reward, done, info1, info2;

    def render(self):
        self.main_body.env.render();


