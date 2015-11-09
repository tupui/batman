# parameters space
space = {
# Lower and upper end points (corners) that define a portion of space.
# format : 2-tuple of tuples with end points coordinates.
   'corners'         : ((0.0, 0.55),
                         (2.0, 0.75),),
# Maximum number of point, used for pod automatic resampling
# format : integer
    'size_max'  : 20,
# Points provider
# Could be a list of points or a dictionary with sampling parameters
    'provider' : {
    # Method used to generate the points
    # format : one of 'uniform', 'halton', 'sobol', 'lhcc', 'lhcr'
        'method' : 'sobol',
    # Number of samples to be generated
    # format : integer
        'size'   : 4,
    }
}

# import functions

snapshot = {
# Snapshot provider
# Could be a python function, a python list of directories or a python dictionary with settings for using an external program like submitting elsA jobs (see snapshot_provider for examples).
#    'provider' : functions.partial(functions.f1, 5),
# Maximum number of simultaneous running snapshot provider
# format : integer > 0
    'max_workers' : 1,
# Input output settings
    'io' : {
    # Names of the parameters
    # format : list of strings
        'parameter_names': ['alpha', 'mach'],
    # File format
    # format : one of 'fmt_tp', 'numpy'
        'format' : 'fmt_tp',
    # File names for each mpi cpu
    # When ran on only 1 cpu, all filenames are gathered
    # format : list of strings
        'filenames' : {0: ['pressure_ex.dat']},
    # Name of the file that contains the coordinates
    # of a point in the space of parameters
        'point_filename' : 'header.py',
    # Directory to store io templates
        'template_directory' : None,
    # Names of the variables contained in a snapshot
    # format : list of strings
        'variables' : ['x','p'],
    # Shapes of one variable for each file and each mpi cpu
    # When ran on only 1 cpu, all shapes are gathered
        'shapes' : {0: [(65,)]},
    },
}


pod = {
# Tolerance of the modes to be kept.
# A percentage of the sum of the singular values, values that account for less than of this tolerance are ignored.
# format : float
    'tolerance' : 0.99,
# Maximum number of modes to be kept
# format : integer
    'dim_max'   : 100,
# Type of pod to perform.
# format : one of 'static', 'dynamic', 'auto'
    'type'      : 'static',
# Stopping criterion for automatic resampling
# format : float
    'quality'   : 0.001*1.e-300,
# Server settings
# None means no server, the pod processing is run from the main python interpreter
    'server' : None,
# Otherwise the pod processing is run in another process or on another computer, the settings is a dictionary with the following parameters.
    # 'server' : {
    #     # Server hostname with port
    #         'port'   : 8000,
    #     # Python executable that runs the server
    #         'python' : 'python',
    #     },
}

prediction = {
# Method used to generate a snapshot
# format : one of 'rbf' , 'kriging'
    'method' : 'rbf',
# Set of points at which the predictions are made
# format : list of tuples of floats
    'points' : [(1.5, 0.6), (2.,0.75)],
}


import numpy as N
num = 10
x = N.linspace(space['corners'][0][0], space['corners'][1][0], num=num)
y = N.linspace(space['corners'][0][1], space['corners'][1][1], num=num)
xy = []
for i in x:
    for j in y:
        xy += [(float(i),float(j))]
#prediction['points'] = xy
