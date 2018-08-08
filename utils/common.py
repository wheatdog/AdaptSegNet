import os
import shutil
import sys

def mkdir_check(path):
    if os.path.exists(path):
        print('{} exists. Do you want to remove the existing directory and proceed? [type n to stop]'.format(path), file=sys.stderr)
        answer = input()
        if answer == 'n':
            exit()
        shutil.rmtree(path)
    os.makedirs(path)
