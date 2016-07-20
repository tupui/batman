import os
import sys
import unittest
import filecmp
import base_test_class

# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.insert(0, path)

from tecplot import *


class TecplotAsciiTest(base_test_class.BaseTest):


    def setUp(self):
        self.io = TecplotAscii()
        self.extension = '.dat'
        bad = {
        # bad names
        'bad-0' : NameError,
        # bad indices
        'bad-1' : ShapeError,
        # bad format
        'bad-2' : FormatError,
        }
        self.bad.update(bad)


    def compare_files(self, file1, file2):
        return filecmp.cmp(file1, file2)




if __name__ == '__main__':
    unittest.main()
