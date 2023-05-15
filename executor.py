# @Time   : 2023.05.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import os, logging
from lib import glb_var
from lib.callback import Logger

if __name__ == '__main__':
    if not os.path.exists("./cache/logger"):
        os.makedirs("./cache/logger");
    glb_var.__init__();
    log = Logger(
        level = logging.DEBUG,
        filename = './cache/logger/logger.log',
    ).get_log()
    glb_var.set_value('logger', log);
