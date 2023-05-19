# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from agent.memory.onpolicy import *
from lib import glb_var, callback

def get_memory(mmy_cfg):
    '''Obtain the corresponding memory according to the configuration
    '''
    if mmy_cfg['name'].lower() in ['onpolicy', 'onpolicymemory']:
        return OnPolicyMemory(mmy_cfg);
    else:
        glb_var.get_value('logger').error(f'Type of memory [{mmy_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');