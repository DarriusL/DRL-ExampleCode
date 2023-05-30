# @Time   : 2023.05.16
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import json_util, util, glb_var, callback
from room.system import *
import torch, time, os

def run_work(cfg_path, mode):
    '''Run the corresponding mode according to the configuration file

    Parameters:
    -----------
    cfg_path:str
        The path of the configuration file
    
    mode:str
    '''
    lab_cfg = json_util.jsonload('./config/lab_cfg.json');
    glb_var.set_values(lab_cfg['constant'])
    glb_var.set_value('mode', mode);
    #parameter report dict
    glb_var.set_value('var_reporter', callback.VarReporter());

    #set device
    if lab_cfg['general']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    else:
        device = torch.device("cpu");
    glb_var.set_value('device', device);
    #report device
    glb_var.get_value('var_reporter').add('device', device);
    #load config
    cfg = json_util.jsonload(cfg_path);
    if cfg['model_path'] is not None:
        cfg['model_path'], _ = os.path.split(cfg_path);
        cfg['model_path'] += '/alg.model';
    #generate system
    system = get_system(cfg);

    if mode == 'train':
        run_train(system);
    elif mode == 'test':
        system.test();

def run_train(system):
    t = time.time();
    glb_var.get_value('logger').info('Start training ... ');
    system.train();
    glb_var.get_value('logger').info(f'Training complete, time consuming: {util.s2hms(time.time() - t)}');