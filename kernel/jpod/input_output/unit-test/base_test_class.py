import os
import sys
import unittest
import filecmp
import fnmatch
import numpy as n

# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.insert(0, path)

from dataset import *
from base import *

# nose must not test this file
__test__ = False

indices = (2,2,2)
d = n.arange(8, dtype=n.float).reshape(indices)
data = n.array([d,-d,d*8])
names = ['x', 'y', 'z']
dataset = Dataset(names=names, data=data)


class BaseTest(unittest.TestCase):

    # dict of bad files to exceptions
    # default one check for no file
    bad = {'bad-filename': IOError}


    def setUp(self):
        self.io = None
        self.extension = None


    def tearDown(self):
        for f in fnmatch.filter(os.listdir('.'), 'new-*'):
            os.remove(f)


    def compare_files(self, file1, file2):
        raise NotImplemented('must be overriden.')


    def file_path(self, filename):
        return os.path.join(os.path.dirname(__file__), filename+self.extension)


    def test_read(self):
        for f,e in self.bad.items():
            self.assertRaises(e, self.io.read, self.file_path(f))

        # read all names
        d = self.io.read(self.file_path('good-0'))
        for name in d.names:
            # check names
            self.assertTrue(name in names)
            # check data
            self.assertTrue(n.array_equal(d[name], dataset[name]))

        # read one bad variable
        self.assertRaises(NameError, self.io.read, self.file_path('good-0'),
                          names=['w'])

        # read one variable at a time, in reversed order
        v = names[:]
        v.reverse()
        for i,v in enumerate(v):
            d = self.io.read(self.file_path('good-0'), names=[v])
            self.assertTrue(n.array_equal(d[v], data[names.index(v)]))


    def test_write(self):
        # check write with good names order
        self.io.write(self.file_path('new-1'), dataset)
        self.assertTrue(self.compare_files(self.file_path('new-1'),
                                           self.file_path('good-0')))
