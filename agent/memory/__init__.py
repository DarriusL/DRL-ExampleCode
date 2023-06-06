# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.onpolicy import *
from agent.memory.offpolicy import *
from lib import glb_var, callback

logger = glb_var.get_value('log');
__all__ = ['get_memory', 'get_batch_split']

def get_memory(mmy_cfg):
    '''Obtain the corresponding memory according to the configuration
    '''
    if mmy_cfg['name'].lower() in ['onpolicy', 'onpolicymemory']:
        return OnPolicyMemory(mmy_cfg);
    elif mmy_cfg['name'].lower() in ['onpolicybatch', 'onpolicybatchmemory']:
        return OnPolicyBatchMemory(mmy_cfg);
    elif mmy_cfg['name'].lower() in ['offpolicy', 'offpolicymemory']:
        return OffPolicyMemory(mmy_cfg);
    elif mmy_cfg['name'].lower() in ['per', 'prioritizedmemory']:
        return PrioritizedMemory(mmy_cfg);
    else:
        logger.error(f'Type of memory [{mmy_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');

def get_batch_split(type):
    #TODO:complete it
    pass

def batch_seq_split(batch, subbatch_num, add_origin = False):
    '''Serialize a batch subserial
    
    Parameters:
    ----------
    batch:dict

    subbatch_num:int

    add_origin:bool,optional
        Whether to add the original batch
        default:False
    '''
    batch_len = len(batch['rewards']);
    subbatch_len = int(batch_len/subbatch_num);
    idxs = torch.from_numpy(np.random.choice(batch_len - subbatch_len + 1, subbatch_num, replace = False));
    subbatchs = []
    for i in range(subbatch_num):
        subbatch = {};
        for key, value in batch.items():
            subbatch[key] = value[idxs[i]:idxs[i]+subbatch_len];
        subbatchs.append(subbatch);

    if add_origin:
        subbatchs.append(batch);
    return subbatchs

def batch_random_split(batch, subbatch_num, add_origin = False):
    #TODO:complete it
    pass
