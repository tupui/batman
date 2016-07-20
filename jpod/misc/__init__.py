from .module_import import import_file
import os

def clean_path(path):
    """Return an absolute and normalized path."""
    return os.path.abspath(os.path.normpath(path))

__all__ = ['import_file', 'clean_path']
