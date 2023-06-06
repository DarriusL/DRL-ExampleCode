# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import os, logging, argparse
from lib import glb_var, callback
from lib.callback import Logger

#TODO:Add: A mode that can be trained on existing models
#TODO:Notes on each algorithm
if __name__ == '__main__':
    if not os.path.exists("./cache/logger"):
        os.makedirs("./cache/logger");
    glb_var.__init__();

    #set arg
    parse = argparse.ArgumentParser();
    parse.add_argument('--config', '-cfg', type = str, default = None, help = 'Path of configration.');
    parse.add_argument('--mode', type = str, default = 'train', help = 'Mode of operation.(train/test)');
    parse.add_argument('--dev', type = bool, default = False, help = 'Enable code debugging');

    args = parse.parse_args();

    log_level = logging.DEBUG if args.dev else logging.INFO;
    log = Logger(
        level = log_level,
        filename = './cache/logger/logger.log',
    )
    logger = log.get_log();
    glb_var.set_value('log', logger);
    glb_var.set_value('dev', args.dev);
    from room.work import run_work
    if args.config is not None and args.mode in ['train', 'test']:
        run_work(args.config, args.mode);
    else:
        logger.error(f'Mode [{args.mode}] is not supported.')
        raise callback.CustomException('ModeError');
