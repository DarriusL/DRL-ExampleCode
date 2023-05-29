# @Time   : 2023.05.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.base import Memory
from agent.memory.onpolicy import OnPolicyMemory
from collections import deque
from lib import glb_var
import numpy as np
from operator import itemgetter
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
    '''Sum Tree algorithm for PER
    '''
    def __init__(self, capacity) -> None:
        self.capacity = capacity;
        self.check_capacity();
        self.write_idx = 0;
        self.tree_data = np.zeros(2 * self.capacity - 1);
        self.node_data = np.zeros(self.capacity, dtype = np.int32);
        
    def check_capacity(self):
        '''Must be a full binary tree, that is, the capacity must be an integer power of 2'''
        capacity = round(self.capacity);
        if capacity & (capacity - 1) == 0:
            self.capacity = capacity;
            return;
        glb_var.get_value('logger').warn('The tree capacity needs to be an integer power of 2, '
                                         'it will be automatically adjusted or canceled to reset');
        #Calculates the nearest power of 2 to the capacity
        power_bin_capacity = len(bin(capacity)) - 2;
        power_bin = np.array([power_bin_capacity - 1, power_bin_capacity, power_bin_capacity + 1]);
        capacity_near = 2 ** power_bin;
        self.capacity = capacity_near[np.argmin(np.abs(capacity_near - capacity))];
        glb_var.get_value('logger').info(f'The tree capacity will be reset to {self.capacity}');
    
    def get_latest_idx(self):
        '''Get the tree index of the last added data'''
        node_idx = (self.write_idx - 1)%self.capacity;
        tree_idx = node_idx + self.capacity - 1;
        return tree_idx;

    def get_root(self):
        '''Get the total value of the tree'''
        return self.tree_data[0];
    
    def get_capacity(self):
        '''Get the capacity of the tree'''
        return self.capacity;

    def _propagate(self, idx, change):
        '''Pass updates to parent node
        
        Parameters:
        -----------
        idx:int
            tree idx

        change:float
            Changed value of tree node 
        '''
        parent_idx = (idx - 1) // 2;
        self.tree_data[parent_idx] += change;
        if(parent_idx != 0):
            self._propagate(parent_idx, change);
    
    def node_update(self, idx, td):
        '''node-level update'''
        change = td - self.tree_data[idx];
        self.tree_data[idx] = td;
        self._propagate(idx, change);

    def add_node(self, nd, td, update_tree = True):
        '''Add a single node to the tree'''
        idx = self.write_idx + self.capacity - 1;
        self.node_data[self.write_idx] = nd;
        self.write_idx = (self.write_idx + 1)%self.capacity;
        #if set to update every single time
        if update_tree:
            change = td - self.tree_data[idx];
            self._propagate(idx, change);
        self.tree_data[idx] = td;
    
    def add_nodes(self, nds, tds, update_tree = True):
        '''Add nodes to the tree

        Parameters:
        -----------
        nds:list

        tds:list
        '''
        assert len(nds) == len(tds);
        num = len(nds);
        nds = np.array(nds);
        tds = np.array(tds);
        write_idxs = np.arange(self.write_idx, self.write_idx + num, dtype = np.int32);
        write_idxs[write_idxs >= self.capacity] -= self.capacity;
        tree_write_idxs = write_idxs + self.capacity - 1;
        self.node_data[write_idxs] = nds;
        self.tree_data[tree_write_idxs] = tds;
        if update_tree:
            self._update_tree();
        self.write_idx = (self.write_idx + num)%self.capacity;

    def _retrieve(self, idx, td_ref):
        '''retrieve tree idx'''
        idx_left = idx * 2 + 1;
        idx_right = idx_left + 1;
        if idx_left >= len(self.tree_data):
            return idx;

        if td_ref <= self.tree_data[idx_left]:
            return self._retrieve(idx_left, td_ref);
        else:
            return self._retrieve(idx_right, td_ref - self.tree_data[idx_left]);

    def fetch(self, td_ref):
        '''retrieve api'''
        if td_ref > self.get_root():
            return None;
        tree_read_idx = self._retrieve(0, td_ref);
        return self.node_data[tree_read_idx - self.capacity + 1], tree_read_idx;

    def _update_tree(self):
        '''Update the whole tree'''
        def __half(a):
            return np.arange(a[0] // 2, a[0] // 2 + 0.5*len(a), dtype = np.int32);
        def __near_2_add(data):
            return data[0:len(data):2] + data[1:len(data):2];

        #update tree
        times = len(bin(self.capacity)) - 3;
        idxs = np.arange(self.capacity - 1, 2 * self.capacity - 1, dtype = np.int32);
        idxs_parent = __half(idxs);
        for _ in range(times):
            self.tree_data[idxs_parent] = __near_2_add(self.tree_data[idxs]);
            idxs = idxs_parent;
            idxs_parent = __half(idxs);

class PrioritizedMemory(Memory):
    '''Prioritized Experience Replay'''
    def __init__(self, memory_cfg) -> None:
        super().__init__(memory_cfg);
        self.is_onpolicy = False;
        #Existing data in memory
        self.is_episodic_exp = False;
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities'];
        self.valid_mmy = OnPolicyMemory({'name':'OnPolicy'});
        self.batch_idxs = None;
        self.tree_idxs = None;
        self.is_training = True;
        self.write_idx = 0;
        self.stock = 0;
        self.reset();

    def train(self):
        '''switch to train'''
        self.is_training = True;
    
    def valid(self):
        '''switch to valid'''
        self.is_training = False;

    def reset(self):
        '''Reset(Clear) the memory'''
        self.sumtree = SumTree(self.max_size);
        self.is_training = True;
        self.write_idx = 0;
        self.stock = 0;
        self.exp_latest = [None] * 6;
        #priority, idx
        self.exps_latest = [];
        for key in self.exp_keys:
            setattr(self, key, [None]*self.max_size);
    
    def _cal_priority(self, omega):
        return (np.abs(omega) + self.epsilon) ** self.eta;

    def update(self, state, action, reward, next_state, done, omega = 1e+5):
        '''Add experience to the memory'''
        if self.is_training:
            self.stock = min(self.stock + 1, self.max_size);
            #cal priority
            #Here we temporarily do not store the priority and idx in the tree
            #After the current round of data acquisition is completed (sampling means that the acquisition is completed), 
            #it will be stored together
            priority = self._cal_priority(omega)
            #add experience
            self.exp_latest = [state, action, reward, next_state, done, priority];
            self.exps_latest.append([self.write_idx, priority]);
            for idx, key in enumerate(self.exp_keys):
                getattr(self, key)[self.write_idx] = self.exp_latest[idx];
            self.write_idx = (self.write_idx + 1)%self.max_size;
        else:
            self.valid_mmy.update(state, action, reward, next_state, done);
    
    def sample_idxs(self):
        self.batch_idxs = [None] * self.batch_size;
        self.tree_idxs = [None] * self.batch_size;
        for idx in range(self.batch_size):
            td_ref = random.uniform(0, self.sumtree.get_root());
            self.batch_idxs[idx], self.tree_idxs[idx] = self.sumtree.fetch(td_ref);

        if self.sample_add_latest:
            self.batch_idxs[-1] = (self.write_idx - 1)%self.max_size;
            self.tree_idxs[-1] = self.sumtree.get_latest_idx();
    
    def _batch_to_tensor(self, batch):
        '''Convert a batch to a format for torch training
        batch['states]:[T, in_dim]
        batch['actions']:[T]
        batch['rewards']:[T]
        batch['next_states']:[T, in_dim]
        '''
        for key in batch.keys():
            batch[key] = torch.from_numpy(np.array(batch[key])).to(glb_var.get_value('device'));
        return batch;


    def get_stock(self):
        '''Returns the amount of experience currently stored'''
        return self.stock;

    def sample(self):
        '''Sample batch'''
        if self.is_training:
            if len(self.exps_latest) != 0:
                #Add the experience that has not been added to the tree first.
                idxs, priorities = tuple(zip(*self.exps_latest));
                self.exps_latest.clear();
                self.sumtree.add_nodes(idxs, priorities, update_tree = True);
            #sample by priorities
            self.sample_idxs();
            batch = {};
            for key in self.exp_keys[:-1]:
                batch[key] = np.array(itemgetter(*self.batch_idxs)(getattr(self, key)));
            return self._batch_to_tensor(batch);

        else:
            return self.valid_mmy.sample();

    def update_priorities(self, omegas):
        '''Update the priority of sampled data
        
        Parameters:
        ----------
        omegas:list
            Requirement:
            1.The length must be consistent with the batch size of the current sampling training.
            2.The order must be consistent with the order in the batch of the current sampling training.
        '''
        priorities = self._cal_priority(omegas);
        for batch_idx, tree_idx, priority in zip(self.batch_idxs, self.tree_idxs, priorities):
            self.priorities[batch_idx] = priority;
            self.sumtree.node_update(tree_idx, priority);


