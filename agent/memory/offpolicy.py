# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.base import Memory
from lib import glb_var
import numpy as np
import torch

class OffPolicyMemory(Memory):
    '''Memory for off policy algorithm, experience is stored according to batch'''
    def __init__(self, memory_cfg) -> None:
        super().__init__(memory_cfg);
        #Existing data in memory
        self.stock = 0;
        self.is_episodic_exp = False;
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones'];
        #the index for latest experience, -1 for empty
        self.idx = -1;
        self.reset();

    def reset(self):
        '''Reset(Clear) the memory'''
        for key in self.exp_keys:
            setattr(self, key, [None] * self.max_size);
        self.exp_latest = [None] * 5;
        self.stock = 0;
        self.idx = -1;

    def update(self, state, action, reward, next_state, done):
        '''Add experience to the memory'''
        self.idx = (self.idx + 1) % self.max_size;
        self.exp_latest = [state, action, reward, next_state, done];
        for idx, key in enumerate(self.exp_keys):
            getattr(self, key)[self.idx] = self.exp_latest[idx];
        if self.stock < self.max_size:
            self.stock += 1;
    
    def _batch_to_tensor(self, batch):
        '''Convert a batch to a format for torch training
        batch['states]:[batch_size, in_dim]
        batch['actions']:[batch_size]
        batch['rewards']:[batch_size]
        batch['next_states']:[batch_size, in_dim]
        batch['dones']"[batch_size]
        '''
        for key in batch.keys():
            batch[key] = torch.from_numpy(np.array(batch[key])).to(glb_var.get_value('device'));
        return batch;


    def sample(self):
        '''sample data'''
        batch = {};
        batch_idxs = np.random.randint(0, self.stock, size = (self.batch_size));
        for key in self.exp_keys:
            batch[key] = np.array(getattr(self, key))[batch_idxs];
        return self._batch_to_tensor(batch)