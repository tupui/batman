# Input output settings for snapshot
io = {
    # Names of the parameters
    # format : list of strings
        'parameter_names': ['x', 'y'],
    # File format
    # format : one of 'fmt_tp', 'numpy'
        'format' : 'numpy',
    # File names
    # TODO: unix patterns
    # format : list of strings
        'filenames' : {0: ['dummy.npz']},
    # name of the file that contains the coordinates
    # of a point in the space of parameters
        'point_filename' : 'header',
    # directory to store io templates
        'template_directory' : None,
    # names of the variables contained in a snapshot
    # format : list of strings
        'variables' : ['dummy'],
    # shapes of one variable for each files
        'shapes' : {0: [(5,)]},
}

# 1D
io_2 = dict(io)
io_2['shapes']    = {0: [[3]          , [2]]}
io_2['filenames'] = {0: ['dummy.0.npz', 'dummy.1.npz']}

# 1D MPI
io_mpi = dict(io)
io_mpi['shapes']    = {0: [[3]]          , 1: [[2]]}
io_mpi['filenames'] = {0: ['dummy.0.npz'], 1: ['dummy.1.npz']}
