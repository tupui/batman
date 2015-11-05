"""Import a python module from a file."""
import os
import sys
import imp

def import_file(full_path_to_module):
    '''Import a file as a module'''
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    # if module_dir == '': module_dir = '.'
    sys.path.insert(0, module_dir)
    return imp.load_source(module_name, full_path_to_module)
