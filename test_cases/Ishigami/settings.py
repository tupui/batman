# parameters space
space = {
# Lower and upper end points (corners) that define a portion of space.
# format : 2-tuple of tuples with end points coordinates.
   'corners'         : ((-3.1415, -3.1415, -3.1415),
                        (3.1415, 3.1415, 3.1415),),
   'delta_space'     : 0.01,                         
# Maximum number of point, used for pod automatic resampling
# format : integer
    'size_max' : 7 ,
# Points provider
# Could be a list of points or a dictionary with sampling parameters
    'provider' : {
    # Method used to generate the points
    # format : one of 'uniform', 'halton', 'sobol', 'lhcc', 'lhcr'
        'method' : 'halton',
    # Number of samples to be generated
    # format : integer
        'size' : 6 ,
    }
}

# import functions

snapshot = {
# Snapshot provider
# Maximum number of simultaneous running snapshot provider
    'max_workers' : 50,
# Input output settings
    'io' : {
    # Names of the parameters
    # format : list of strings
        'parameter_names': ['x1','x2', 'x3'],
    # File format
    # format : one of 'fmt_tp', 'numpy'
        'format' : 'fmt_tp',
    # File names for each mpi cpu
    # When ran on only 1 cpu, all filenames are gathered
    # format : list of strings
        'filenames' : {0: ['function.dat']},
    # Name of the file that contains the coordinates
    # of a point in the space of parameters
        'point_filename' : 'header.py',
    # Directory to store io templates
        'template_directory' : None,
    # Names of the variables contained in a snapshot
    # format : list of strings
        'variables' : ['F'],
    # Shapes of one variable for each file and each mpi cpu
    # When ran on only 1 cpu, all shapes are gathered
        'shapes' : {0: [(1,)]},
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
# format : one of 'static', 'dynamic'
    'type'      : 'static',
# Resampling strategy: None, 'MSE', 'loo_mse', 'loo_sobol', 'extrema', 'hybrid'
    'resample'  : 'MSE',
    'strategy' : (('MSE', 2), ('loo_sobol', 0), ('extrema', 1)),
# Stopping criterion for resampling
# format : float
    'quality'   : 0.80,            
# Server settings
# None means no server, the pod processing is run from the main python interpreter
    'server' : None,
# Otherwise the pod processing is run in another process or on another computer, the settings is a dictionary with the following parameters.
    # 'server' : {
    #     # Server hostname with port
    #         'port'   : 800199,
    #     # Python executable that runs the server
    #         'python' : 'python',
    #     },
}

prediction = {
# Method used to generate a snapshot
# format : one of 'rbf' , 'kriging'
    'method' : 'kriging',
# Set of points at which the predictions are made
# format : list of tuples of floats
    'points' : [],
}

uq = {
# Type of Sobol analysis: 'sobol', 'FAST' (if FAST, no second-order indices)
    'method' : 'sobol',
# Type of indices we want: 'aggregated', 'block'
    'type' : 'aggregated',
# Use a test method: 'Ishigami'
    'test' : 'Ishigami',
    'sample' : 1000 ,
# Uncertainty propagation. Enter the PDF of the inputs. x1: Normal(mu, sigma), x2: Uniform(inf, sup)
    'pdf' : ['Uniform(-3.1415, 3.1415)', 'Uniform(-3.1415, 3.1415)', 'Uniform(-3.1415, 3.1415)']
}

import numpy as np
import itertools
num = 25
x = np.linspace(space['corners'][0][0], space['corners'][1][0], num=num)
y = np.linspace(space['corners'][0][1], space['corners'][1][1], num=num)
z = np.linspace(space['corners'][0][1], space['corners'][1][1], num=num)
xyz = []
for i, j, k in itertools.product(x, y, z):
    xyz += [(float(i), float(j), float(k))]
    prediction['points'] = xyz
