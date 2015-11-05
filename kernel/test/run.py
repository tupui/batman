exe_seq = '/data/softs/local/python2.7/bin/python'
exe_mpi = 'mpirun -n 2 python'
jpod    = '../jpod/ui.py'

import os
import sys
from optparse import OptionParser
from numpydiff import numpydiff
from ordereddict import OrderedDict
opj = os.path.join


import subprocess
try:
    # for python 2.7 and above
    check_output = subprocess.check_output
    CalledProcessError = subprocess.CalledProcessError
except:
    # for python 2.6 and below
    # this is a raw copy of 2.7 code but the docstring
    class CalledProcessError(Exception):
        def __init__(self, returncode, cmd, output=None):
            self.returncode = returncode
            self.cmd = cmd
            self.output = output
        def __str__(self):
            return "Command '%s' returned non-zero exit status %d" % (self.cmd, self.returncode)
    
    def check_output(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise CalledProcessError(retcode, cmd, output=output)
        return output


def run(options):
    """docstring for run"""
    # aliases
    output    = options.output
    reference = options.reference

    print output 
    print reference

    # define the tests
    tests = OrderedDict()
#   tests['static_sobol'] = {
#       'compare' : opj(reference, 'static_sobol', 'OUT')}
#    tests['static_sobol_points'] = {
#         'compare' : opj(output, 'static_sobol')}
#    tests['static_sobol_server'] = {
#         'compare' : opj(output, 'static_sobol')}
#    tests['static_sobol_files'] = {
#         'compare' : opj(output, 'static_sobol')}
#    tests['static_sobol_server_task_0'] = {
#         'compare' : opj(output, 'static_sobol')}
    tests['static_sobol_task_0'] = {
         'compare' : opj(output, 'static_sobol')}
#   tests['static_sobol_server_task_1'] = {
#       'compare' : opj(output, 'static_sobol')}
#   tests['static_halton'] = {
#       'compare' : opj(reference, 'static_halton', 'OUT')}
#   tests['static_lhcc'] = {
#       'compare' : opj(reference, 'static_lhcc', 'OUT')}
#   tests['static_lhcr'] = {
#       'compare' : opj(reference, 'static_lhcr', 'OUT')}
#   tests['dynamic_sobol'] = {
#       'compare' : opj(reference, 'dynamic_sobol', 'OUT')}
#   tests['dynamic_sobol_server'] = {
#       'compare' : opj(output, 'dynamic_sobol')}
#   tests['dynamic_sobol_files'] = {
#       'compare' : opj(output, 'dynamic_sobol')}
#    tests['dynamic_sobol_mpi'] = {
#        'compare' : opj(reference, 'dynamic_sobol_mpi', 'OUT')}
#    tests['dynamic_sobol_task0'] = {
#        'compare' : opj(output, 'dynamic_sobol_mpi')}
#    tests['dynamic_sobol_mpi_server'] = {
#        'compare' : opj(output, 'dynamic_sobol_mpi')}
#   tests['auto'] = {
#       'compare' : opj(reference, 'auto', 'OUT')}
#   tests['auto_server'] = {
#       'compare' : opj(output, 'auto')}
#   tests['auto_files'] = {
#       'compare' : opj(output, 'auto')}
#    tests['auto_mpi'] = {
#        'compare' : opj(reference, 'auto_mpi', 'OUT')}
#    tests['auto_mpi_server'] = {
#        'compare' : opj(output, 'auto_mpi')}

    # common command line options
    common_cl_options = ['-s']
    if options.restart:
        common_cl_options += ['-r']
    elif options.no_pod:
        common_cl_options += ['-n']

    # run the tests
    for t, v in tests.items():
        if t.endswith('mpi'):
            cmd = exe_mpi.split()
        else:
            cmd = exe_seq.split()

        out = opj(output, t) + v.get('suffix', '')
        cmd += [jpod, opj(options.input, t)+'.py', '-o', out]
        cmd += common_cl_options
        cmd += v.get('options', [])

        # get terminal columns number
        try:
            rows, columns = check_output('stty size'.split()).split()
            columns = int(columns)
        except CalledProcessError:
            columns = 80

        # print what's being run
        print '~' * columns
        print 'executing : ' + ' '.join(cmd)

        # redirect process output
        if options.verbose:
            stderr = None
        else:
            stderr = subprocess.STDOUT

        # run test in a subprocess and catch output
        try:
            stdoutput = check_output(cmd, stderr=stderr)
        except CalledProcessError:
            raise Exception('Test failed, use -v to see output')

#       # compare results to reference
#       if v['compare'] is None:
#           continue
#       elif v['compare'].endswith('OUT'):
#           ref = v['compare']
#       else:
#           ref = opj(v['compare'], 'predictions')
#
#       # predictions
#       print '\tPredictions'
#       numpydiff(ref, opj(out, 'predictions'), recursive=True,
#                  format='npz', verbosity=0, include=['*Newsnap*.npz'])
#
#       if v['compare'].endswith('OUT'):
#           ref = v['compare']
#       else:
#           ref = opj(v['compare'], 'pod')
#
#        # Mean
#        print '\tPOD Mean'
#        numpydiff(opj(ref, 'Mean'), opj(out, 'pod', 'Mean'),
#                  recursive=True, format='npz', verbosity=0)
#        # Modes
#        print '\tPOD Modes'
#        numpydiff(ref, opj(out, 'pod'), include=['*Mod_*.npz'],
#                  recursive=True, format='npz', verbosity=0)
#        # pod
#        print '\tPOD misc'
#        numpydiff(opj(ref, 'pod.npz'), opj(out, 'pod', 'pod.npz'),
#                  format='npz', verbosity=0)


def main(argv=None):
    """Parse and check options, and then call XXX()."""

    if argv is None:
        argv = sys.argv #[1:]

    parser = OptionParser()

    # command line options
    parser.add_option(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='verbosity level.')

    parser.add_option(
        '-o', '--output',
        default='output',
        help='path to output directory.')

    parser.add_option(
        '-r', '--restart',
        action='store_true',
        default=False,
        help='restart all tests.')

    parser.add_option(
        '-n', '--no-pod',
        action='store_true',
        default=False,
        help='do not compute pod but read it from disk, [default: %default].')

    parser.add_option(
        '--reference',
        default='/data/home/elsa/jjouhaud/JPOD_AD/kernel/test/output',
        help='path to reference directory.')

    parser.add_option(
        '-i', '--input',
        default='scripts',
        help='path to input directory.')

    # parse command line
    (options, args) = parser.parse_args()

    run(options)


if __name__ == "__main__":
    main()
