import os
import sys
import unittest
import filecmp

# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.insert(0, path)

from point import PointBase, Point, DimensionError

names  = ('x', 'y', 'z')
values = (0., 1., 2.)
threshold = 1.

local_path = os.path.dirname(__file__)
def abs_path(filename):
    return os.path.join(local_path, filename)


class PointBaseTest(unittest.TestCase):


    def test_set_threshold(self):
        self.assertRaises(ValueError, PointBase.set_threshold, None)
        self.assertRaises(ValueError, PointBase.set_threshold, -1)


    def test__eq__(self):
        PointBase.set_threshold(threshold)
        self.assertEqual(PointBase((0,1,2)), PointBase((0,1,2+1)))
        self.assertNotEqual(PointBase((0,1,2)), PointBase((0,1,2+1.1)))




class PointTest(unittest.TestCase):


    def tearDown(self):
        # remove created stuff
        for f in ['point-2']:
            if os.path.exists(abs_path(f)):
                os.remove(abs_path(f))


    def test_set_names(self):
        # reset names
        Point.names = None
        # empty names
        self.assertRaises(ValueError, Point.set_names, [])
        # bad type
        self.assertRaises(TypeError, Point.set_names, [0])
        Point.set_names(names)
        self.assertEqual(Point.names, names)
        # different names
        self.assertRaises(ValueError, Point.set_names, ['x', 'x', 'x'])
        # different dimensions
        self.assertRaises(ValueError, Point.set_names, ['x'])


    def test__init__(self):
        Point.set_names(names)
        self.assertEqual(Point((0,1,2)), values)
        self.assertEqual(Point([0,1,2]), values)
        # bad value type
        self.assertRaises(ValueError, Point, coordinates=[0,'a',2])
        # different dimensions
        self.assertRaises(DimensionError, Point, coordinates=[0])


    def test_read(self):
        # bad path
        self.assertRaises(IOError, Point, '/')
        self.assertRaises(TypeError, Point, None)
        # good one
        self.assertEqual(Point(abs_path('point-1')), values)
        # bad file
        self.assertRaises(ValueError, Point, abs_path('bad-point'))


    def test_write(self):
        Point(values).write(abs_path('point-2'))
        self.assertTrue(filecmp.cmp(abs_path('point-1'), abs_path('point-2')))




if __name__ == '__main__':
    unittest.main()
