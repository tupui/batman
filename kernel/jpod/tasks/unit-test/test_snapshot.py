import os
import sys
import unittest
import filecmp
import fnmatch
import shutil

opj = os.path.join

# get the testee importable
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../..')
if path not in sys.path:
    sys.path.insert(0, path)

from tasks import SnapshotTask, TaskTimeoutError, TaskFailed
from space import Point
from pod import Snapshot


context = 'context-dir'
data_files = ['output/test-file']
command = 'bash'
script = {}
script['ok']     = os.path.join(context, 'script-dir', 'script_ok.sh')
script['fail-0'] = os.path.join(context, 'script-dir', 'script_fail_0.sh')
script['fail-1'] = os.path.join(context, 'script-dir', 'script_fail_1.sh')
timeout = 1
private_directory = 'private-directory'
initialize_args = [context, command, script['ok'], timeout, data_files,
        private_directory]

Point.set_names(('x', 'y'))
point = Point((0,0))
working_directory = 'working-directory'

snapshot_io_settings = {
    'format' : 'numpy',
    'filenames' : {0: ['dummy.npz']},
    'point_filename' : 'header', # we actually solely need that one
    'template_directory' : None,
    'variables' : ['dummy'],
    'shapes' : {0: [(1,)]},
}


class SnapshotTaskTest(unittest.TestCase):


    def setUp(self):
        Snapshot.initialize(snapshot_io_settings)
        SnapshotTask.initialize(*initialize_args)


    def tearDown(self):
        SnapshotTask._reset()
        if os.path.isdir(working_directory):
            shutil.rmtree(working_directory)


    def test_class_settings(self):
        # class attributes not initialized
        SnapshotTask._reset()
        self.assertRaises(Exception, SnapshotTask, None, None)

        bad = ['test_snapshot.py',
               None,
               'context-dir',
               [None, 0],
               ['toto', [None]],
               None]

        for i, p in enumerate(initialize_args):
            args = initialize_args[:]
            if isinstance(bad[i], list):
                tests = bad[i]
            else:
                tests = [bad[i]]
            for t in tests:
                args[i] = t
                self.assertRaises(ValueError, SnapshotTask.initialize, *args)


    def test_init(self):
        # bad init args
        self.assertRaises(TypeError, SnapshotTask, point, None)
        self.assertRaises(TypeError, SnapshotTask, (0,0), working_directory)


    def test_before_run(self):
        st = SnapshotTask(point, working_directory)

        # test existing working dir
        os.makedirs(working_directory)
        self.assertRaises(TaskFailed, st._before_run)
        os.rmdir(working_directory)

        st._before_run()

        # check point file
        p = Point(opj(working_directory, private_directory)
        self.assertEqual(p, point)

        # check hooked script file
        ref_script = opj('working-directory-ref', 'private-directory', 'script_ok.sh')
        new_script = opj('working-directory', 'private-directory', 'script_ok.sh')
        self.assertTrue(filecmp.cmp(new_script, ref_script))


    def test_copytree_with_symlinks(self):
        opa = os.path.abspath
        st = SnapshotTask(point, working_directory)
        st._copytree_with_symlinks(opa(context), opa(working_directory))
        for root, dirs, files in os.walk(opa(context)):
            local = root.replace(context, working_directory)
            for d in dirs:
                self.assertTrue(os.path.isdir(opj(local, d)))
            for f in files:
                self.assertEqual(opj(root, f), os.path.realpath(opj(local, f)))


    def test_run_fail_0(self):
        # missing finished state file
        initialize_args_ = initialize_args[:]
        initialize_args_[2] = script['fail-0']
        SnapshotTask.initialize(*initialize_args_)
        st = SnapshotTask(point, working_directory)
        self.assertRaises(TaskTimeoutError, st.run)


    def test_run_fail_1(self):
        # missing snapshot file state file
        initialize_args_ = initialize_args[:]
        initialize_args_[2] = script['fail-1']
        SnapshotTask.initialize(*initialize_args_)
        st = SnapshotTask(point, working_directory)
        self.assertRaises(TaskFailed, st.run)


    def test_run(self):
        st = SnapshotTask(point, working_directory)
        self.assertEqual(st.run(), opj(os.getcwd(), working_directory,
                         private_directory))




if __name__ == '__main__':
    unittest.main()
