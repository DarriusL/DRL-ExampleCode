# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import json_util, util, glb_var
from room.system import System
import torch, time

def run_work(cfg_path, mode):
    lab_cfg = json_util.jsonload('./config/lab_cfg.json');
    glb_var.set_values(lab_cfg['constant'])
    #set device
    if lab_cfg['general']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    else:
        device = torch.device("cpu");
    glb_var.set_value('device', device);
    #load config
    cfg = json_util.jsonload(cfg_path);
    #generate system
    system = System(cfg);

    if mode == 'train':
        run_train(system);

def run_train(system):
    t = time.time();
    glb_var.get_value('logger').info('Start training ... ');
    system.train();
    glb_var.get_value('logger').info(f'Training complete, time consuming: {util.s2hms(time.time() - t)}');