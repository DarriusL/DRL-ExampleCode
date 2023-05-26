# @Time   : 2023.05.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.base import Memory
from agent.memory.onpolicy import OnPolicyMemory
from collections import deque;
from lib import glb_var, util
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
        self.repository = deque(maxlen = self.max_size);
        self.valid_mmy = OnPolicyMemory({'name':'OnPolicy'});
        self.is_training = True;
        self.reset();
    
    def train(self):
        '''switch to train'''
        self.is_training = True;
    
    def valid(self):
        '''switch to valid'''
        self.is_training = False;

    def reset(self):
        '''Reset(Clear) the memory'''
        self.repository.clear();
        self.exp_latest = [None] * 5;


    def update(self, state, action, reward, next_state, done):
        '''Add experience to the memory'''
        if self.is_training:
            self.exp_latest = [state, action, reward, next_state, done];
            self.repository.append(copy.deepcopy(self.exp_latest));
        else:
            self.valid_mmy.update(state, action, reward, next_state, done);
    
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

    def get_stock(self):
        '''Returns the amount of experience currently stored'''
        return len(self.repository);

    def sample(self):
        '''sample data'''
        if self.is_training:
            batch_sample = random.sample(self.repository, self.batch_size);
            if self.sample_add_latest:
                #Add latest experience
                batch_sample[-1] = self.exp_latest;
            batch_sample = tuple(zip(*batch_sample));
            batch = {}
            for idx, key in enumerate(self.exp_keys):
                batch[key] = np.array(batch_sample[idx]);
            return self._batch_to_tensor(batch);
        else:
            return self.valid_mmy.sample();

class SumTree():
    ''''''
    def __init__(self, sumtree_cfg) -> None:
        util.set_attr(self, sumtree_cfg);
        self.check_capacity();
        self.stock = 0;
        self.write_idx = 0;
        self.tree_data = np.zeros(2 * self.capacity - 1);
        self.node_data = np.zeros(self.capacity);
        
    def check_capacity(self):
        #Must be a full binary tree, that is, the capacity must be an integer power of 2
        capacity = round(self.capacity);
        if capacity & (capacity - 1) == 0:
            self.capacity = capacity;
            return;
        glb_var.get_value('logger').warn('The tree capacity needs to be an integer power of 2, '
                                         'it will be automatically adjusted or canceled to reset');
        power_bin_capacity = len(bin(capacity)) - 2;
        power_bin = np.array([power_bin_capacity - 1, power_bin_capacity, power_bin_capacity + 1]);
        capacity_near = 2 ** power_bin;
        self.capacity = capacity_near[np.argmin(np.abs(capacity_near - capacity))];
        glb_var.get_value('logger').info(f'The tree capacity will be reset to {self.capacity}');

    def get_root(self):
        ''''''
        return self.NodeData[0];
    
    def get_capacity(self):
        ''''''
        return self.capacity;

    def _propagate(self, idx, change):
        ''''''
        # Pass updates to parent node
        parent_idx = (idx - 1) // 2;
        self.tree_data[parent_idx] += change;
        if(parent_idx != 0):
            self._propagate(parent_idx, change);

    def add(self, nd, td):
        ''''''
        idx = self.write_idx + self.capacity - 1;
        self.node_data[self.write_idx] = nd;
        self.write_idx += 1;
        if self.write_idx >= self.capacity:
            self.write_idx = 0;
        #if set to update every single time
        if self.single_update:
            change = td - self.tree_data[idx];
            self._propagate(idx, change);
        self.tree_data[idx] = td;

    def _retrieve(self, idx, td_ref):
        ''''''
        idx_left = idx * 2 + 1;
        idx_right = idx_left + 1;
        if idx_left >= len(self.tree_data):
            return idx;

        if td_ref <= self.tree_data[idx_left]:
            return self.retrieve_sgnode(idx_left, td_ref);
        else:
            return self.retrieve_sgnode(idx_right, td_ref - self.tree_data[idx_left]);

    def fetch(self, td_ref):
        ''''''
        if td_ref > self.get_root():
            return None;
        read_idx = self._retrieve(0, td_ref);
        return self.node_data[read_idx];

    def update_tree(self):
        ''''''
        def __half(a):
            return np.arange(a[0] // 2, a[0] // 2 + 0.5*len(a), dtype = np.int64);
        def __near_2_add(data):
            return data[0:len(data):2] + data[1:len(data):2];
        if not self.single_update:
            #update tree
            times = len(bin(self.capacity)) - 3;
            idxs = np.arange(self.capacity - 1, 2 * self.capacity - 1, dtype = np.int64);
            idxs_parent = __half(idxs);
            for _ in range(times):
                self.tree_data[idxs_parent] = __near_2_add(self.tree_data[idxs]);
                idxs = idxs_parent;
                idxs_parent = __half(idxs);


