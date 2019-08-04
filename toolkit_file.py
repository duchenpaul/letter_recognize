import os, re
import os.path
import glob
from pathlib import Path
import codecs
import shutil

import logging

def check_file_exists(FILE):
    '''Check if the FILE exists'''
    return Path(FILE).is_file()


def check_dir_exists(DIR):
    '''Check if the DIR exists'''
    return Path(DIR).is_dir()


def get_basename(FILE, withExtension=False):
    '''
    Return the basename of a file. e.g. example.txt -> example
    '''
    if withExtension:
        return ''.join(os.path.splitext(os.path.basename(FILE)))
    else :
        return os.path.splitext(os.path.basename(FILE))[0]


def file_path(FILE):
    '''Return the path of the folder where the FILE locates'''
    return os.path.dirname(os.path.realpath(FILE)) + os.sep


def script_path():
    '''Return the path of the current script'''
    return os.path.dirname(os.path.realpath(__file__))


def line_prepender(filename, line):
    '''
    Add line to the head of a file
    '''
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def get_file_list(folder):
    '''Return the list of file of the folder'''
    file_list = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            file_list.append(os.path.join(path, name))
    return file_list


def purge_folder(folder, filePattern='*'):
    '''Empty everything in the folder'''
    # filelist = [ f for f in os.listdir(folder) ] #if f.endswith(".bak") ]
    filelist = glob.glob(folder + os.sep + filePattern)
    for f in filelist:
        # print(f)
        os.remove(os.path.join(f)) # using glob
        # os.remove(os.path.join(folder, f)) # using listdir


def create_folder(folderName):
    '''Create folder if not exists'''
    my_file = Path(folderName)
    if not my_file.is_dir():
        logging.info('Folder {} not found, creating a new one'.format(folderName))
        os.mkdir(folderName)


def text_replace_in_file(pattern, string, file):
    '''Replace pattern with string in file'''
    with open(file) as f:
        replaced_script = re.sub(pattern, string, f.read(), flags=re.IGNORECASE)
    with open(file, 'w') as f:
        f.write(replaced_script)


def convert_encode2utf8(sourceFileName, targetFileName, srcEncoding = 'utf-16'):
    '''Save the file with encoding of utf-8'''
    BLOCKSIZE = 1048576 # or some other, desired size in bytes
    with codecs.open(sourceFileName, 'r', 'utf-16') as sourceFile:
        with codecs.open(targetFileName, 'w', 'utf-8') as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)


def remove_junk_line(FILE, junkwords):
    '''
    Remove the line that contains junkwords
    '''
    with open(FILE) as oldfile, open(FILE + 'tmp', 'w') as newfile:
        for line in oldfile:
            if not junkwords in line:
                newfile.write(line)
    shutil.move(FILE + 'tmp', FILE)

if __name__ == '__main__':
    print(file_path(r'C:\Users\chdu\Downloads\sf'))
