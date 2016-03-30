import os
import sys
import unittest
import filecmp

# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../..')
if path not in sys.path:
    sys.path.insert(0, path)

from space import *
from point import PointBase

values = (0., 1., 2.)
names  = ('x', 'y')
corners = ((0, 0), (1, 1))
max_points_nb = 100


local_path = os.path.dirname(__file__)
def abs_path(filename):
    return os.path.join(local_path, filename)


class SpaceBaseTest(unittest.TestCase):


    def tearDown(self):
        # remove created stuff
        if os.path.exists(abs_path('space')):
            os.remove(abs_path('space'))


    def test_add(self):
        PointBase.set_threshold(1.)
        s = SpaceBase()
        s.add(PointBase((0,0)))
        self.assertRaises(UnicityError, s.add , PointBase((0,0)))
        # check nothing happens
        self.assertEqual(s.add(PointBase((0,1.1))), None)


    def test_dim(self):
        s = SpaceBase()
        s.add(PointBase((0,0)))
        self.assertEqual(s.dim, 2)


    def test_size(self):
        s = SpaceBase()
        self.assertEqual(s.size, 0)
        s.add(PointBase((0,0)))
        self.assertEqual(s.size, 1)


    def test_io(self):
        s = SpaceBase()
        s.add(PointBase((0,0)))
        path = abs_path('space')
        s.write(path)
        ss = SpaceBase()
        ss.read(path)
        self.assertEqual(s, ss)




class SpaceTest(unittest.TestCase):


    def test__init__(self):
        self.assertRaises(TypeError, Space, (0,), names, max_points_nb)
        self.assertRaises(DimensionError, Space, ((0, 0), (1,)), names,
                          max_points_nb)
        self.assertRaises(ValueError, Space, ((0, 0), (0, 1)), names,
                          max_points_nb)


    def test_add(self):
        s = Space(corners, names, 0.0, 1)
        self.assertRaises(AlienPointError, s.add, [(0,2)])
        s.add([(0,0)])
        self.assertRaises(UnicityError, s.add , [(0,0)])
        self.assertRaises(FullSpaceError, s.add , [(0,1)])




if __name__ == '__main__':
    unittest.main()
