import os
import sys
import unittest
import numpy as n
import base_test_class
from base_test_class import names, dataset, data


# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.insert(0, path)

from npz import *


class NpzTest(base_test_class.BaseTest):


    def setUp(self):
        self.io = Npz()
        self.extension = '.npz'


    def compare_files(self, file1, file2):
        d1 = dict(n.load(file1))
        d2 = dict(n.load(file2))
        if d1.keys() != d2.keys():
            return False
        else:
            for k in d1:
                if not n.array_equal(d1[k], d2[k]):
                    return False
        return True




if __name__ == '__main__':
    unittest.main()
