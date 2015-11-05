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

shape = (2,2,2)
d = n.arange(8, dtype=n.float).reshape(shape)
data = n.array([d,-d,d*8])
names = ('x', 'y', 'z')
names3 = ('z', 'x', 'y')

class DatasetInfoTest(unittest.TestCase):


    def setUp(self):
        self.info = DatasetInfo(names=names, shape=shape)


    def test_names(self):
        self.assertRaises(NameError, self.info.set_names, 0)
        self.assertRaises(NameError, self.info.set_names, '0')
        self.assertRaises(NameError, self.info.set_names, [])
        self.assertRaises(NameError, self.info.set_names, [0])


    def test_shape(self):
        self.assertRaises(ShapeError, self.info.set_shape, 0)
        self.assertRaises(ShapeError, self.info.set_shape, '0')
        self.assertRaises(ShapeError, self.info.set_shape, [])
        self.assertRaises(ShapeError, self.info.set_shape, ['0'])


    def test_eq(self):
        info2 = DatasetInfo()
        info2.set_names(names)
        info2.set_shape(shape)
        info3 = DatasetInfo(names=names3, shape=shape)
        self.assertEqual(self.info, info2)
        self.assertNotEqual(self.info, info3)


    def test_size(self):
        self.assertEqual(self.info.size, 3*8)




if __name__ == '__main__':
    unittest.main()
