.. _settings:

Settings
========


Introduction
^^^^^^^^^^^^


| A detailed analysis about ``settings_template.py`` - the most important file in the directory ``scripts`` - will be presented in this document

| In this file, there are five blocks with different functions:


Block 1 - Parameters space
^^^^^^^^^^^^^^^^^^^^^^^^^^


| In this block, we create the coordinate with the sampling points and chose the method to generate these points.

.. code-block:: python

    space = {'corners'     : ((1., 1.), (3.1415, 3.1415),),
              'delta_space' : 0.01,                         
              'size_max'    : 21,
              'provider'    : {'method' : 'halton',
                               'size'   : 20,
                              }
              }

+ | ``corners``: Define a portion of space.
+ | ``delta_space``: Make an additional exterior space with the selected space for taking sample points in the borders.
+ | ``size_max``: Maximum number of point for POD automatic resampling.
+ | ``method``: One of method in OpenTURNS such as: *uniform*, *halton*, *sobol*, *lhcc* (Latin Hypercube Centered) or *lhcr* (Latin Hypercube Random).
+ | ``size``: Number of samples to be generated.


Some useful information
"""""""""""""""""""""""

+ | In *probability theory and statistics*, a **probability density function (PDF)** is used to represent a probability in the form of integrals.
+ | **Uniform distribution**: Depending on the nature of the PDF, this distribution can be *continuous* or *discrete*. In the case of *Continuous uniform distribution*, according to the probability distribution, all of the intervals with the same length have the same probability. Due to the shape of the PDF, it also known as *Rectangular distribution*.
+ | **Halton distribution**: It is constructed according to a *deterministic method* that uses coprime numbers as its bases.
+ | **Sobol distribution**: It is an example of *quasi-random* (or *low-discrepancy*) sequence in the range from 0 to 1.
+ | **Latin Hypercube distribution**: It is a statistical method for generating a near-random sample of parameter values from a multidimensional distribution. In particular, the sampling in each grid is *centered* or *random* corresponding to *LHCC* or *LHCR*.

|

Block 2 - Snapshot provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^


| It could be *a python function*, *a python list of directories* or *a python dictionary* with settings for using *an external program* like submitting *elsA* jobs.

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

+ | ``max_workers``: Maximum number of simultaneous running snapshot provider.
+ | ``parameter_names``: Names of the parameters.
+ | ``format``:  *fmt_tp* (Tecplot 360) or *numpy* (NumPy).
+ | ``filenames``: For each MPI CPU. When ran on only 1 CPU, all filenames are gathered.
+ | ``point_filename``: Name of the file that contains the coordinates of a point in the space of parameters.
+ | ``template_directory``: Directory to store Input/Output templates.
+ | ``variables``: Names of the variables contained in a snapshot.
+ | ``shapes``: Shapes of 1 variable for each file and each MPI CPU. When ran on only 1 CPU, all shapes are gathered.


Some useful information
"""""""""""""""""""""""

+ | **Message Passing Interface (MPI)** is a standardized and portable message-passing system designed by a group of researchers from academia and industry to function on a wide variety of *parallel computers*. For more information, please click `here <http://www.mpi-forum.org/>`_!!!
+ | **Tecplot 360** is a visual data analysis tool that improves your productivity with integrated XY, 2D, and 3D plotting.

|

Block 3 - POD
^^^^^^^^^^^^^


| POD (or Proper Orthogonal Decomposition) is a approach to help reduce amount of data.

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

+ | ``tolerance``: Tolerance of the modes to be kept. A percentage of the sum of the singular values, values that account for less than of this tolerance are ignored.
+ | ``dim_max``: Maximum number of modes to be kept.
+ | ``type``: Type of POD to perform: *static*, *dynamic* or *auto*.
+ | ``resample``: Type of resampling strategy: *None*, *MSE* (*Mean Squared Error*), *loo_mse* (*Leave-one-out* integrates *Mean Squared Error*), *loo_sobol* (*Leave-one-out* integrates *Sobol sequence*), *extrema* or *hybrid*. Moreover, the priority order is evaluated from left to right.
+ | ``strategy``: **Only** meaningful in which case ``resample`` is *hybrid*.
+ | ``quality``: Stopping criterion for automatic resampling. In this example, if the value of error from approximating the surrogate model > 0,8 then it will not do the resampling. 
+ | ``server``: Server settings. *None* means **no server**, the POD processing is run from the main python interpreter.


Some useful information
"""""""""""""""""""""""

1. | *Mean Squared Error (MSE)* of an estimator measures the average of the squares of the errors or deviations (so it also known as *Mean Squared Deviation (MSD)*). In other words, it means the difference between the estimator and what is estimated: :math:`MSE=\frac{1}{n} \sum_{i=1}^n (Y_i^{\hat} - Y_i)^2`.

2. | *Leave-one-out (LOO)*: Assume that we are given a set of points in a space (for example, a surface).

    + | *Firstly*, we start by taking one data point out of this set.
    
    + | *Secondly*, we train a classifier with the same algorithm but without this point.
    
    + | *Thirdly*, we test the classifier on this point.
    
    + | To complete the procedure, we repeat these steps for all the data points.
    
    + | *In short*, compute the LOO estimate as the *sum of the errors* divided by the *number of data*.


3. | *Extrema*: i.e. *maxima* (or *largest value*) and *minima* (or *smallest value*) of a function.

    + | When these values can be achieved on *a given range* of a function, we have the *local* (or *relative*) extrema.
    
    + | In the case that they are on the *entire domain* of a function, they called the *global* (or *absolute*) extrema.


4. | *Quatity*: i.e. Predictive squared correlation coefficient: :math:`Q^2=1-\frac{\sum_{i=1}^n (Y_i^{\hat} - Y_i)^2}{\sum_{i=1}^n (Y_i^{\tilde} - Y_i)^2}=1-\frac{n\cdot MSE}{\sum_{i=1}^n (Y_i^{\tilde} - Y_i)^2}`.

|

Block 4 - Prediction
^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    prediction = {'method' : 'kriging',
                  'points' : [ ],
                 }

+ | ``method``: Method used to generate a snapshot one of *rbf* (or *Radial Basic Function*) or *kriging* method (or *KGM*).
+ | ``points``: Set of points at which the predictions are made.
+ | We can fill *directly* the number of points into the brackets or *indirectly* via the script.


Some useful information
"""""""""""""""""""""""

1. | The *RBF* is a real-valued function whose value depends only on the distance from the origin, so that: :math:`\phi(x)=\phi(||x||)`.

2. | The *KGM* is a statistical prediction of a function at *untried inputs*. KGM is a flexible and robust technique to build fast *surrogate models* based on small experimental designs.

|

Block 5 - UQ
^^^^^^^^^^^^


| UQ (or *Uncertainty Quantification*) is used as a method to evaluate the results.

.. code-block:: python

    uq = {'method' : 'sobol',
          'type'   : 'aggregated',
          'sample' : 5000 ,
          'pdf'    : ['Uniform(-2.048, 2.048)',
                      'Uniform(-2.048, 2.048)']
         }

+ | ``method``: Type of Sobol analysis: *sobol*, *FAST* (or *Fourier Amplitude Sensitivity Testing*) (if FAST, no second-order indices).
+ | ``type``: Type of indices we want: *aggregated* or *block*.
+ | ``sample``: Use a test method: *Ishigami*.
+ | ``pdf`` (or *Probability density function*): Uncertainty propagation. Enter the PDF of the inputs: x1: Normal(mu, sigma), x2: Uniform(inf, sup).


Some useful information
"""""""""""""""""""""""

1. | The *FAST* is a variance-based global sensitivity analysis method. The sensitivity value is defined based on conditional variances which indicate the individual or joint effects of the uncertain inputs on the output.

2. | The *Ishigami function* of Ishigami & Homma (1990) is used as an example for uncertainty and sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity. For more information, please visit the `Derek Bingham website <http://www.sfu.ca/~ssurjano/ishigami.html>`_!!!
Theme:  Basic  Nature Save with unique link 
Quick reStructuredText referenceCopyright © rst.ninjs.org, 2011