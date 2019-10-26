import os
from . import __file__

__all__ = ['full_path']


def full_path(path):
    return os.path.join(os.path.dirname(__file__), path)
