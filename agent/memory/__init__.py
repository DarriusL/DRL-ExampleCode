# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.onpolicy import *
from agent.memory.offpolicy import *
from lib import glb_var, callback

logger = glb_var.get_value('log');

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