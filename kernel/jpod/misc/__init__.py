from module_import import import_file
from logging_conf import setup
import os

def clean_path(path):
    """Return an absolute and normalized path."""
    return os.path.abspath(os.path.normpath(path))

try:
    # available from python 2.7
    from collections import OrderedDict
except ImportError:
    # back port for python 2.5-2.6
    from ordereddict import OrderedDict