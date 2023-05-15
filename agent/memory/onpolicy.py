# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.base import Memory

class OnPolicyMemory(Memory):
    def __init__(self, memory_cfg) -> None:
        super().__init__(memory_cfg);
        #Existing data in memory
        self.stock = 0;
        #Suitable for episodic environments
        self.is_episodic_exp = True;
        #Experience data that needs to be stored
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones'];

    def reset(self):
        return super().reset()