# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import os, logging, argparse
from lib import glb_var
from lib.callback import Logger
from room.work import run_work

if __name__ == '__main__':
    if not os.path.exists("./cache/logger"):
        os.makedirs("./cache/logger");
    glb_var.__init__();
    log = Logger(
        level = logging.DEBUG,
        filename = './cache/logger/logger.log',
    ).get_log()
    glb_var.set_value('logger', log);
    #set arg
    parse = argparse.ArgumentParser();
    parse.add_argument('--config', '-cfg', type = str, default = None, help = 'Path of configration.');
    parse.add_argument('--mode', type = str, default = 'train', help = 'Mode of operation.');

    args = parse.parse_args();

    if args.config is not None:
        run_work(args.config, args.mode);
