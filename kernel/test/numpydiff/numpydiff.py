#!/usr/bin/env python
import os
import sys
from fnmatch import fnmatch
from optparse import OptionParser, Values
import numpy as N

# force numpy to raise an exception on floating-point errors
# N.seterr(all='raise')

path = os.path.dirname(os.path.realpath(__file__))
if path not in sys.path:
    sys.path.insert(0, path)

try:
    import _elsa_io as elsa
except:
    print 'cannot import _elsa_io'
    pass

__all__ = ['numpydiff']
__version__ = 1.1
__docformat__ = "reStructuredText"

help_message = '''numpydiff [options] arg1 arg2

where arg1 and arg2 can be files or directories.'''

# default files to exclude
exclude_default = ['.*']
include_default = []


def file_reader(format, path, variables):
    """Dispatch data reading according to `format`."""
    if format == 'npz':
        lazy_data = N.load(path)
        data = dict(lazy_data)
        if variables:
            for v in variables:
                if v not in data:
                    data.pop(v)
#        lazy_data.close()

    elif format in ['fmt_tp', 'fmt_v3d', 'bin_v3d']:
        # elsa IO likes to play tricks ... like changing variable names!
        # so we have to revert them back
        elsa_io_variable_map = {
        "rou"  : "rovx",
        "rov"  : "rovy",
        "row"  : "rovz",
        "psta" : "p"   ,
        "pgen" : "Pi"  ,
        "roE"  : "roe" ,
        }

        names, data = elsa.readAll(path, format)
        data = dict(zip(names, data))

        # elsaIO variables picking is totally screwed up,
        # so we have to read all variables and filter afterwards
        if variables:
            for v in data.keys():
                v_ = elsa_io_variable_map.get(v, v)
                if v_ in variables:
                    data[v_] = data.pop(v)
                else:
                    data.pop(v)

    else:
        raise IOError('bad file format.')

    return data

def error_header():
    return '%-15s %-15s %-15s'%('Variable', 'Absolute error', 'Relative error')


def error_string(variable, error):
    return '%-15s %-15g %-15g'%(variable, error['abs'], error['rel'])


def match(path, patterns):
    """Checks whether a file at `path` must be ignored or not."""
    for p in patterns:
        if fnmatch(path, p):
            return True
    return False


def diff_array(ref, new):
    if ref.size != new.size:
        raise Exception('data sizes are different !')
    diff = ref - new
    error = {}
    error['abs'] = N.amax(N.fabs(diff))
    if error['abs'] != 0.:
        try:
            error['rel'] = N.amax(N.fabs(diff / ref))
        except FloatingPointError:
            pass
    else:
        error['rel'] = 0.

    return error


def diff_file(ref, new, options):
    if options.verbosity > 0:
        print 'Comparing ', ref
        print 'to        ', new

    ref_data = file_reader(options.format, ref, options.variables)
    new_data = file_reader(options.format, new, options.variables)

    errors = {}
    for v in ref_data:
        errors[v] = diff_array(ref_data[v], new_data[v])
        if options.verbosity > 0:
            print error_string(v, errors[v])

    return errors


def diff_dir(ref, new, options):
    errors_max = {}

    for f in os.listdir(ref):
        ref_file = os.path.join(ref, f)

        if not match(ref_file, options.include) \
           or match(ref_file, options.exclude):
            continue

        rel_path = ref[len(ref)+1:]
        new_file = os.path.join(new, rel_path, f)
        errors = diff_file(ref_file, new_file, options)

        # compute max errors
        for v in errors:
            if not errors_max.has_key(v):
                errors_max[v] = {'abs' : 0., 'rel' : 0.}

            for e in errors[v]:
                if errors[v][e] > errors_max[v][e]:
                    errors_max[v][e] = errors[v][e]
                    errors_max[v]['file'] = ref_file
    return errors_max


def diff(ref, new, options):
    '''Compute the numeric differences between 2 files or set of files.

    :param ref: path to first file
    :param new: path to second file
    :param options: object with options as attributes
    '''
    # clean paths
    ref = os.path.normpath(ref)
    new = os.path.normpath(new)

    print error_header()

    if os.path.isfile(ref):
        errors = diff_file(ref, new, options)

    elif options.recursive:
        errors_max = {}
        for path, dirs, files in os.walk(ref):
            for f in files:
                rel_path = os.path.join(path[len(ref)+1:], f)
                if not match(rel_path, options.include) \
                   or match(rel_path, options.exclude):
                    continue
                ref_file = os.path.join(ref, rel_path)
                new_file = os.path.join(new, rel_path)
                errors = diff_file(ref_file, new_file, options)
                # compute max errors
                for v in errors:
                    if v not in errors_max:
                        errors_max[v] = {'abs' : 0., 'rel' : 0.}

                    for e in errors[v]:
                        if errors[v][e] > errors_max[v][e]:
                            errors_max[v][e] = errors[v][e]
                            errors_max[v]['file'] = ref_file
        errors = errors_max
    else:
        errors = diff_dir(ref, new, options)

    if options.verbosity >= 0 :
        if options.verbosity > 0:
            print '\n', 'Error summary:'
        for v in errors:
            print error_string(v, errors[v])
            if errors[v].has_key('file') \
               and options.show_file:
                print '\t file  :', errors[v]['file']

    return errors


def numpydiff(ref, new,
              recursive=False,
              exclude  =exclude_default,
              include  =['*'],
              format   ='bin_v3d',
              show_file=False,
              variables=None,
              verbosity=0):
    '''Python interface to numpydiff

    See OptionParser for arguments info.
    '''
    options = Values()
    options.recursive = recursive
    options.exclude   = exclude
    options.include   = include
    options.format    = format
    options.variables = variables
    options.verbosity = verbosity
    options.show_file = show_file
    return diff(ref, new, options)




def main(argv=None):
    """Parse and check options, and then call diff."""

    if argv is None:
        argv = sys.argv #[1:]

    parser = OptionParser(usage=help_message, version=__version__)

    parser.add_option('-r', '--recursive',
        action='store_true',
        dest='recursive',
        default=False,
        help='Recursively compare any subdirectories found.')

    parser.add_option('-i', '--include',
        action='append',
        dest='include',
        default=include_default,
        metavar='PAT',
        help='Include files that match the pattern PAT.')

    parser.add_option('-x', '--exclude',
        action='append',
        dest='exclude',
        default=exclude_default,
        metavar='PAT',
        help='Exclude files that match the pattern PAT.')

    parser.add_option('-f', '--format',
        dest='format',
        default='bin_v3d',
        metavar='FORMAT',
        help='File format: fmt_tp, fmt_v3d, bin_v3d, npz [default: %default]')

    parser.add_option('-V', '--variables',
        dest='variables',
        default=None,
        metavar='VAR',
        help='Compare variables VAR, [default: compare all variables]')

    parser.add_option('-v',
        action='count',
        dest='verbosity',
        default=0,
        help='Verbose mode, [default: %default]')

    parser.add_option('--show-file',
        dest='show_file',
        action='store_true',
        default=False,
        help='show most differing file, [default: %default]')

    (options, args) = parser.parse_args()
    if options.include == []:
        options.include = ['*']

    if len(args) != 2:
        parser.error("incorrect number of arguments")

    # do the diff
    try:
        diff(args[0], args[1], options)
        return 0
    except:
        raise
        return 1




if __name__ == "__main__":
    sys.exit(main())
