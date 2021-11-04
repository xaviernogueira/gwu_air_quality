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

def move_or_delete_files(in_folder, out_folder, str_in):
    """
    Moves files from one folder to another by a string query
    :param in_folder: Folder containing files to be moved
    :param out_folder: Folder to move files to, if 'False', files will be deleted!!
    :param str_in: string that if is contained within a file path, the file is selected to move
    """
    import shutil

    # initialize logger and find files
    init_logger(__file__)
    all_files = os.listdir(in_folder)
    move_names = []

    for file in all_files:
        if str_in in file:
            move_names.append(file)

    move_dirs = [(in_folder + '\\%s' % i) for i in move_names]

    if isinstance(out_folder, str):
        # make out_folder if it doesn't exist already
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # create output locations and move files
        out_dirs = [(out_folder + '\\%s' % i) for i in move_names]

        for i, dir in enumerate(move_dirs):
            shutil.move(dir, out_dirs[i])
            logging.info('Moved %s from %s to %s' % (move_names[i], in_folder, out_folder))

        return out_folder

    elif isinstance(out_folder, bool) and not out_folder:
        # take user input to list of files to be deleted
        print('STOP SIGN: Check list of files that will be deletes, enter Y to proceed and N to stop!')
        print('Files to delete: %s ' % move_names)
        val = input('Input Y to proceed:')

        if val:
            for dir in move_dirs:
                try:
                    os.remove(dir)
                    logging.info('Deleted %s' % dir)
                except PermissionError:
                    print('Could not get permission to delete %s' % dir)

            return print('Deleted %s files in %s' % (len(move_dirs), in_folder))

    else:
        return print('out_folder parameter must be a directory name to move files, or =False to delete files')


#################################################

dems = r'C:\Users\xrnogueira\Documents\Data\3DEP'

