"""
Python 3 functions for using APIs to pull and organize large temporal datasets

@author: xrnogueira
"""
import logging
import subprocess
import os

# static variables
USERNAME = ''  # data hub username
PASSWORD = ''  # data hub password


def init_logger(filename):
    """Initializes logger w/ same name as python file"""

    logging.basicConfig(filename=os.path.basename(filename).replace('.py', '.log'), filemode='w', level=logging.INFO)
    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderr_logger)

    return


def cmd(command):
    """Executes command prompt command and logs messages"""

    logger = logging.getLogger(__name__)
    try:
        res = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        msg = 'Command failed: %s' % command
        logger.error(msg)
        raise Exception(msg)

    # log any cmd line messages
    msg = res.communicate()[1]
    logger.info(msg)

    return
