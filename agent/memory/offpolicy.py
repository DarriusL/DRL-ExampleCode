# @Time   : 2023.05.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.base import Memory
from collections import deque;
from lib import glb_var
import numpy as np
import torch, copy, random

class OffPolicyMemory(Memory):
    '''Memory for off policy algorithm, experience is stored according to batch'''
    def __init__(self, memory_cfg) -> None:
        super().__init__(memory_cfg);
        self.is_onpolicy = False;
        #Existing data in memory
        self.is_episodic_exp = False;
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones'];
        self.repository = deque(maxlen = self.max_size)
        self.reset();

    def reset(self):
        '''Reset(Clear) the memory'''
        self.repository.clear();
        self.exp_latest = [None] * 5;


    def update(self, state, action, reward, next_state, done):
        '''Add experience to the memory'''
        self.exp_latest = [state, action, reward, next_state, done];
        self.repository.append(copy.deepcopy(self.exp_latest));
    
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
        if self.batch_size > len(self.repository):
            return None;
        batch_sample = random.sample(self.repository, self.batch_size);
        if self.sample_add_latest:
            #Add latest experience
            batch_sample[-1] = self.exp_latest;
        batch_sample = zip(*batch_sample);
        batch = {}
        for idx, key in enumerate(self.exp_keys):
            batch[key] = np.array(batch_sample[idx]);

        return self._batch_to_tensor(batch)