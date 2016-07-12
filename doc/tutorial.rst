.. _tutorial:


Tutorial
********

Step by step
============

Your need execute access to the directory ``test-cases`` and create there the test directories. Generally, each test-case module consists of 2 main parts:

+ Directory ``data``: this directory contains files describing the code of ``CFD calculation`` (AVBP, elsA, etc.) or the code of *optimization problems*, for example.

+ Directory ``scripts``: this directory contains python script files which  compiled by users for setting the ``sampling methods``.

|

A simple example
================


We chose the **Michalewicz function** for this test-case. For other examples of optimization problems, you can see in the `website of Derek Bingham <http://www.sfu.ca/~ssurjano/optimization.html>`_ (please contact the author via email: dbingham@stat.sfu.ca).

|

File ``settings_template.py``
-----------------------------


In fact, ``settings_template.py`` is the most important file in the directory `scripts`. In this file, there are 5 different blocks:

|

Block 1 - Parameters space
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this paragraph, we create the coordinate with the sampling points and chose the method to generate these points.

.. code-block:: python

     space = {'corners'     : ((1., 1.), (3.1415, 3.1415),),
              'delta_space' : 0.01,                         
              'size_max'    : 21,
              'provider'    : {'method' : 'halton',
                               'size'   : 20,
                              }
              }

+ ``corners``: Define a portion of space.
+ ``delta_space``: Make an additional exterior space with the selected space for taking sample points in the borders.
+ ``size_max``: Maximum number of point for POD automatic resampling.
+ ``method``: One of method in OpenTURNS such as: **uniform**, **halton**, **sobol**, **lhcc** (Latin Hypercube Centered) or **lhcr** (Latin Hypercube Random).
+ ``size``: Number of samples to be generated.


Some useful information
"""""""""""""""""""""""

+ ``Uniform distribution``: *Continuous uniform distribution* or *Rectangular distribution* is a family of symmetric probability distributions such that for each member of the family, all intervals of the same length on the distribution's support are equally probable.
+ ``Halton distribution``: In statistics, Halton sequences are sequences used to generate points in space for numerical methods such as Monte Carlo simulations.
+ ``Sobol distribution``: In statistics, Sobol sequences are an example of quasi-random low-discrepancy sequences.
+ ``Latin Hypercube distribution``: It is a statistical method for generating a near-random sample of parameter values from a multidimensional distribution. In particular, sampling is *centered* or *random* in each grid corresponding to ``LHCC`` or ``LHCR``.


|

Block 2 - Snapshot provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It could be *a python function*, *a python list of directories* or *a python dictionary* with settings for using *an external program* like submitting ``elsA`` jobs.

.. code-block:: python

     snapshot = {'max_workers' : 50,
                 'io'          : {'parameter_names'    : ['x1','x2'],
                                  'format'             : 'fmt_tp',
                                  'filenames'          : {0: ['function.dat']},
                                  'point_filename'     : 'header.py',
                                  'template_directory' : None,
                                  'variables'          : ['F'],
                                  'shapes'             : {0: [(1,)]},
                                 },
                 }

+ ``max_workers``: Maximum number of simultaneous running snapshot provider.
+ ``parameter_names``: Names of the parameters.
+ ``format``:  **fmt_tp** (Tecplot 360) or **numpy** (NumPy).
+ ``filenames``: For each MPI CPU. When ran on only 1 CPU, all filenames are gathered.
+ ``point_filename``: Name of the file that contains the coordinates of a point in the space of parameters.
+ ``template_directory``: Directory to store Input/Output templates.
+ ``variables``: Names of the variables contained in a snapshot.
+ ``shapes``: Shapes of 1 variable for each file and each MPI CPU. When ran on only 1 CPU, all shapes are gathered.

Some useful information
"""""""""""""""""""""""


+ ``Message Passing Interface (MPI)`` is a standardized and portable message-passing system designed by a group of researchers from academia and industry to function on a wide variety of *parallel computers*. For more information, please click `here <http://www.mpi-forum.org/>`_!!!
+ ``Tecplot 360`` is a visual data analysis tool that improves your productivity with integrated XY, 2D, and 3D plotting.
+ ``NumPy`` is the fundamental package for scientific computing with Python.

|

Block 3 - POD
^^^^^^^^^^^^^

POD (or ``Proper Orthogonal Decomposition``) is a approach to help reduce amount of data.

.. code-block:: python

     pod = {'tolerance' : 0.99,
            'dim_max'   : 100,
            'type'      : 'static',
            'resample'  : 'extrema',
            'strategy'  : (('MSE', 2), ('loo_sobol', 0),
                           ('extrema', 1)),
            'quality'   : 0.8,
            'server'    : None,
           }

+ ``tolerance``: Tolerance of the modes to be kept. A percentage of the sum of the singular values, values that account for less than of this tolerance are ignored.
+ ``dim_max``: Maximum number of modes to be kept.
+ ``type``: Type of POD to perform: **static**, **dynamic** or **auto**.
+ ``resample``: Type of resampling strategy: **None**, **MSE** (*Mean Squared Error*), **loo_mse** (*Leave-one-out* integrates *Mean Squared Error*), **loo_sobol** (*Leave-one-out* integrates *Sobol sequence*), **extrema** or **hybrid**. Moreover, the priority order is evaluated from left to right.
+ ``strategy``: Only meaningful in which case ``resample`` is **hybrid**.
+ ``quality``: Stopping criterion for automatic resampling. In this example, if the value of ``error from approximating the snapshots ???`` > 0,8 then it will not do the resampling. 
+ ``server``: Server settings. **None** means no server, the POD processing is run from the main python interpreter.


Some useful information
"""""""""""""""""""""""

1. In statistics, the ``Mean Squared Error (MSE)`` or ``Mean Squared Deviation (MSD)`` of an estimator measures the average of the squares of the errors or deviations, that is, the difference between the estimator and what is estimated: :math:`MSE=\frac{1}{n} \sum_{i=1}^n (Y_i^{\hat} - Y_i)^2`.

2. ``Leave-one-out (LOO)`` strategy: Assume that we are given a set S of n points in a space (for example, a surface). We start by taking one data point out of this set. Then, we train a classifier with the same algorithm but without this point ... and test the classifier on this point. To complete the procedure, we repeat these steps for all the data points. In short, compute the LOO estimate of the risk as the average error over the whole procedure, i.e., as the sum of the errors divided by the number of data.

3. ``Extrema``

|

Block 4 - Prediction
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

     prediction = {'method' : 'kriging',
                   'points' : [ ],
                  }

+ ``method``: Method used to generate a snapshot one of **rbf** (*Radial Basic Function*) or **kriging**.
+ ``points``: Set of points at which the predictions are made.
+ We can fill *directly* the number of points into the brackets or *indirectly* via the script.


Some useful information
"""""""""""""""""""""""

1. A ``Radial Basis Function (RBF)`` is a real-valued function whose value depends only on the distance from the origin, so that: :math:`\phi(x)=\phi(||x||)`.

2. ``Kriging method (KGM)`` is a statistical prediction of a function at *untried inputs*. KGM is a flexible and robust technique to build fast ``surrogate models`` based on small experimental designs.

|

Block 5 - UQ
^^^^^^^^^^^^

UQ (or ``Uncertainty Quantification``) is used as a method to evaluate the results.

.. code-block:: python

     uq = {'method' : 'sobol',
           'type'   : 'aggregated',
           'sample' : 5000 ,
           'pdf'    : ['Uniform(-2.048, 2.048)',
                       'Uniform(-2.048, 2.048)']
          }

+ ``method``: Type of Sobol analysis: **sobol**, **FAST** (*Fourier Amplitude Sensitivity Testing*) (if FAST, no second-order indices).
+ ``type``: Type of indices we want: **aggregated** or **block**.
+ ``sample``: Use a test method: **Ishigami**.
+ ``pdf`` (or *Probability density function*): Uncertainty propagation. Enter the PDF of the inputs: x1: Normal(mu, sigma), x2: Uniform(inf, sup).


Some useful information
"""""""""""""""""""""""

1. The ``FAST`` is a variance-based global sensitivity analysis method. The sensitivity value is defined based on conditional variances which indicate the individual or joint effects of the uncertain inputs on the output.

2. The ``Ishigami function`` of Ishigami & Homma (1990) is used as an example for uncertainty and sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity. For more information, please visit the `Derek Bingham website <http://www.sfu.ca/~ssurjano/ishigami.html>`_!!!

|

File ``function.py``
--------------------


Beside the ``settings_template.py``, we take the ``function.py`` in the directory **data** to describe the information of **optimization test functions**.

Keep in mind that the test case here is Michalewicz function.

.. code-block:: python

     F = -1.0-math.sin(X1)*(math.pow(math.sin(X1*X1/math.pi),20.))-math.sin(X2)*(math.pow(math.sin(2*X2*X2/math.pi),20.))

|

Remarks
-------

Simplistically, we change only the formula of the function and the coordinate in the optimization problems.


Implementation with POD
=======================
bla bla bla