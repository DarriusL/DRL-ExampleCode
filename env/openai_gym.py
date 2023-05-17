# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from env.base import Env, _make_env


class OpenaiEnv(Env):
    '''the openai environment'''
    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg);
        self.env = _make_env(env_cfg);
    
    def get_state_and_action_dim(self):
        '''(state_dim, action_choice)
        '''
        return self.env.observation_space.shape[0], self.env.action_space.n;

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


