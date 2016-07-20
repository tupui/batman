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


class DatasetTest(unittest.TestCase):


    def setUp(self):
        self.d = Dataset(names=names, data=data)


    def test_set_data(self):
        # set shape from data
        self.assertEqual(self.d.shape, shape)

        # data with bad size
        self.assertRaises(DataSizeError, self.d.set_data, data[0])

        # data with good size but different shape
        data2 = data.ravel()
        self.d.set_data(data2)
        self.assertTrue(n.array_equal(self.d.data, data))

        # check input shape was not modified
        self.assertEqual(data2.ndim, 1)


    def test_getitem(self):
        # bad key
        self.assertRaises(KeyError, self.d.__getitem__, 'toto')

        # check quantity data
        for i,name in enumerate(names):
            self.assertTrue(n.array_equal(self.d[name], data[i]))


    def test_setitem(self):
        # bad key
        self.assertRaises(KeyError, self.d.__setitem__, 'toto', 0)

        # replace data
        data2 = data.copy()
        for i,name in enumerate(names):
            data2[i] = 0
            self.d[name] = n.zeros(8)
            self.assertTrue(n.array_equal(self.d.data, data2))


    def test_info(self):
        info = DatasetInfo(names=names, shape=shape)
        self.assertEqual(info, self.d.info)




if __name__ == '__main__':
    unittest.main()
