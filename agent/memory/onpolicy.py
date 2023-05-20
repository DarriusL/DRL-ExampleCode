# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.base import Memory

class OnPolicyMemory(Memory):
    '''Memory for on policy algorithm in episodic env, experience is stored according to epoch

    Parameters:
    -----------
    memory_cfg:dict
    '''
    def __init__(self, memory_cfg) -> None:
        super().__init__(memory_cfg);
        #Existing data in memory
        self.stock = 0;
        #Suitable for episodic environments
        self.is_episodic_exp = True;
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones'];
        self.reset();

    def reset(self):
        '''Clear all experience storage memory
        '''
        for key in self.exp_keys:
            setattr(self, key, []);
        self.exp_latest = [None] * len(self.exp_keys);
        self.cur_exp = {k: [] for k in self.exp_keys};
        self.stock = 0;

    def update(self, state, action, reward, next_state, done):
        '''Add experience to experience memory

        '''
        self.exp_latest = [state, action, reward, next_state, done];
        for idx, key in enumerate(self.exp_keys):
            self.cur_exp[key].append(self.exp_latest[idx]);
        
        self.stock += 1;
        if done:
            for key in self.exp_keys:
                getattr(self, key).append(self.cur_exp[key]);

    def sample(self):
        '''sample data

        Returns:
        --------
        batch:dict
            e.g.
            batch = {
                'states':  [[s_epoch0], [s_epoch1], [s_epoch2], ...],
                'actions':[[a_epoch0], [a_epoch1], [a_epoch2], ...],
                'rewards':[[r_epoch0], [r_epoch1], [r_epoch2], ...],
                ...
            }
        '''
        batch = {key: getattr(self, key) for key in self.exp_keys};
        #onpolicy
        self.reset();
        return batch;

class OnPolicyBatchMemory(OnPolicyMemory):
    '''Memory for on policy algorithm in episodic env, experience is stored according to batch

    Parameters:
    -----------
    memory_cfg:dict
    '''
    def __init__(self, memory_cfg) -> None:
        super().__init__(memory_cfg);
        self.is_episodic_exp = False;

    def update(self, state, action, reward, next_state, done):
        '''add experience'''
        self.exp_latest = [state, action, reward, next_state, done];
        self.stock += 1;
        for idx, key in enumerate(self.exp_keys):
            getattr(self, key).append(self.exp_latest[idx]);

    def sample(self):
        '''sample data

        Returns:
        --------
        batch:dict
            e.g.
            batch = {
                'states':  [s0, s1, s2, ...],
                'actions':[a0, a1, a2, ...],
                'rewards':[r0, r1, r2, ...],
                ...
            }
        '''
        return super().sample()