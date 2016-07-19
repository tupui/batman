import os
import sys
import unittest
import filecmp
import shutil
import fnmatch
import numpy as n

# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.insert(0, path)
path = os.path.join(path, '..')
if path not in sys.path:
    sys.path.insert(0, path)

from snapshot import *

local_path = os.path.dirname(__file__)
def abs_path(filename):
    return os.path.join(local_path, filename)

indices = (2,2,2)
d = n.arange(8, dtype=n.float).reshape(indices)
data = n.array([d,-d,d*8])
variables = ['x', 'y', 'z']
ref = N.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 0.9999999, 0.9999999])
files = ['good-0.dat', 'good-2.dat']
template_path = abs_path('templates')
point = [1.5, 1.0]

settings = {
'point_filename'     : 'point',
'variables'          : ['y'],
'filenames'          : {0: files},
'template_directory' : template_path,
'format'             : 'fmt_tp',
}


class SnapshotTest(unittest.TestCase):


    def setUp(self):
        Snapshot.initialize(settings)


    def tearDown(self):
        # remove created stuff
        if os.path.exists(template_path):
            shutil.rmtree(template_path)

        for p in ['new*', template_path]:
            for d in fnmatch.filter(os.listdir(local_path), p):
                if os.path.isdir(d):
                    shutil.rmtree(d)

        for f in fnmatch.filter(os.listdir(local_path), 'new-*'):
            os.remove(f)


    def test__create_templates(self):
        # do it once and record modification time
        Snapshot._create_templates(local_path)
        mtime = []
        for f in files:
            template = os.path.join(template_path, f)
            mtime += [os.path.getmtime(template)]
            self.assertTrue(filecmp.cmp(abs_path(f), template))

        # second time, check the files have not been copied again
        Snapshot._create_templates(local_path)
        for i,f in enumerate(files):
            template = os.path.join(template_path, f)
            self.assertEqual(os.path.getmtime(template), mtime[i])

        # check bad directory
        Snapshot.template_directory = files[0]
        self.assertRaises(IOError, Snapshot._create_templates, '.')


    def test_read(self):
        s = Snapshot()
        s.read(local_path)

        # check data
        self.assertTrue(n.array_equal(s.data, ref))

        # check templates are there
        self.assertTrue(os.path.isdir(template_path))


    def test_write(self):
        # with point
        s = Snapshot(point=point, data=ref)

        # from existing templates
        Snapshot.template_directory = local_path
        new_path = abs_path('new')
        s.write(new_path)
        for f in os.listdir(new_path):
            self.assertTrue(filecmp.cmp(f, 'new/'+f)) #WTF ???

        # without point
        s = Snapshot(data=ref)
        new2_path = abs_path('new2')
        s.write(new2_path)
        for f in os.listdir(new2_path):
            self.assertNotEqual(f, 'point')
            self.assertTrue(filecmp.cmp(f, 'new2/'+f)) #WTF ???




if __name__ == '__main__':
    unittest.main()
