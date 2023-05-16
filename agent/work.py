# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import json_util, util, glb_var

def run_work():
    lab_cfg = json_util('./config/lab_cfg.json');
    util.set_attr(glb_var, lab_cfg['constant']);