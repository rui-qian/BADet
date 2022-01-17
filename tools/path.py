from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.abspath(os.path.dirname(__file__))

# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir, './../mmdet')
add_path(lib_path)
lib_path = os.path.join(this_dir, './')
add_path(lib_path)
lib_path = os.path.join(this_dir, './../../pointnet')
add_path(lib_path)
lib_path = os.path.join(this_dir, './../../pointnet/lib')
add_path(lib_path)


if __name__ == '__main__':
    print(lib_path)

