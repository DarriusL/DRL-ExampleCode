from agent.algorithm.reinforce import *
from agent import algorithm
from lib import callback

__all__ = ['get_alg'];

def get_alg(alg_cfg):
    try:
        AlgClass = getattr(algorithm, alg_cfg['name']);
        return AlgClass(alg_cfg);
    except:
        if alg_cfg['name'].lower() == 'reinforce':
            return Reinforce(alg_cfg);
        else:
            glb_var.get_value('logger').error(f'Type of algorithm [{alg_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
            raise callback.CustomException('NetCfgTypeError');