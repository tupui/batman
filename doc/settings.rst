.. _settings:

Settings
========


Introduction
^^^^^^^^^^^^


The file ``settings.py`` contains the configuration of BATMAN. It can is devided into 5 blocks.


Block 1 - Space of Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First of all, we define the space of parameters using an hypercube. Taking the minimal and the maximal value along all coordinates allow to describe it.

.. figure:: fig/hypercube.pdf

   3-dimentionnal hypercube

.. code-block:: python

    "space": {
        "corners": [[1.0, 1.0],[3.1415, 3.1415]],
        "size_max": 15,
        "delta_space": 0.01,
        "provider": {
        "method": "halton",
        "size": 15
        }

+ ``corners``: define the space using corners of the hypercube,
+ ``delta_space``: innerspace defined by corners for resampling,
+ ``size_max``: maximum number of samples to be calculated: initial sampling + resampling,
+ ``method``: method to create the DoE: *uniform*, *faure*, *halton*, *sobol*, *sobolscramble*, *lhs* (Latin Hypercube Sampling) or *lhsc* (Latin Hypercube  Sampling Centered),
+ ``size``: initial size of the DoE.

The method used to create the DoE is paramount. It ensures that that the physics will be captured correclty all over the domain of interest, see :ref:`space`. All *faure*, *halton* and *sobol* methods are low discrepancy sequences with good filling properties.


Block 2 - Snapshot provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^


It could be *a python function*, *a python list of directories* or *a python dictionary* with settings for using *an external program* like submitting *elsA* jobs.

.. code-block:: python

    "snapshot": {
        "max_workers": 10,
        "io": {
        "shapes": {"0": [[1]]},# mklj mlkj mlkj
        "format": "fmt_tp_fortran",
        "variables": ["F"],
        "point_filename": "header.py",
        "filenames": {"0": ["function.dat"]},
        "template_directory": null,
        "parameter_names": ["x1", "x2"]
        }

+ ``max_workers``: maximum number of simultaneous running snapshot provider.
+ ``parameter_names``: names of the parameters.
+ ``format``:  *fmt_tp_fortran* (Tecplot 360) included in BATMAN.
+ ``filenames``: for each MPI CPU. When ran on only 1 CPU, all filenames are gathered.
+ ``point_filename``: name of the file that contains the coordinates of a point in the space of parameters.
+ ``template_directory``: depreciated option                             
+ ``variables``: names of the variables contained in a snapshot.
+ ``shapes``: shapes of 1 variable for each file and each MPI CPU. When ran on only 1 CPU, all shapes are gathered.


Block 3 - POD
^^^^^^^^^^^^^


POD (or Proper Orthogonal Decomposition) is a approach to help reduce amount of data.

.. code-block:: python

     "pod": {
        "dim_max": 100,
        "tolerance": 0.99,
        "resample": "MSE",
        "strategy": [["MSE", 4]],
        "quality": 0.8,
        "server": null,
        "type": "static"
     }

+ ``tolerance``: tolerance of the modes to be kept. A percentage of the sum of the singular values, values that account for less than of this tolerance are ignored.
+ ``dim_max``: maximum number of modes to be kept.
+ ``type``: type of POD to perform: *static*, *dynamic* or *auto*.
+ ``resample``: type of resampling strategy: *None*, *MSE* (*Mean Squared Error*), *loo_mse* (*Leave-one-out* integrates *Mean Squared Error*), *loo_sobol* (*Leave-one-out* integrates *Sobol sequence*), *extrema* or *hybrid*.
+ ``strategy``: **Only** meaningful if ``resample`` is set to *hybrid*.
+ ``quality``: stopping criterion for automatic resampling. Here, if the value of error from approximating the surrogate model > 0,8 then the resampling will be stopped. 
+ ``server``: depreciated option. 


Some useful information
"""""""""""""""""""""""

1. *Mean Squared Error (MSE)* of an estimator measures the average of the squares of the errors or deviations (so it also known as *Mean Squared Deviation (MSD)*). In other words, it means the difference between the estimator and what is estimated: :math:`MSE=\frac{1}{n} \sum_{i=1}^n (Y_i^{\hat} - Y_i)^2`.

2. *Leave-one-out (LOO)*: Assume that we are given a set of points in a space (for example, a surface).

    + *Firstly*, we start by taking one data point out of this set.
    
    + *Secondly*, we train a classifier with the same algorithm but without this point.
    
    + *Thirdly*, we test the classifier on this point.
    
    + To complete the procedure, we repeat these steps for all the data points.
    
    + *In short*, compute the LOO estimate as the *sum of the errors* divided by the *number of data*.


3. *Extrema*: i.e. *maxima*  and *minima* of a function.

    + When these values can be achieved on *a given range* of a function, we have the *local* (or *relative*) extrema.
    
    + In the case that they are on the *entire domain* of a function, they called the *global* (or *absolute*) extrema.


4. *Quatity*: i.e. predictive squared correlation coefficient: :math:`Q^2=1-\frac{\sum_{i=1}^n (Y_i^{\hat} - Y_i)^2}{\sum_{i=1}^n (Y_i^{\tilde} - Y_i)^2}=1-\frac{n\cdot MSE}{\sum_{i=1}^n (Y_i^{\tilde} - Y_i)^2}`.


Block 4 - Prediction
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    "prediction": {
        "points": [],
        "method": "kriging"
    }

+ ``method``: method used to generate a snapshot one of *rbf* (*Radial Basic Function*) or *kriging* (*KGM*) method.
+ ``points``: set of points at which the predictions are made.

.. note:: We can fill *directly* the number of points into the brackets or *indirectly* via the script ``prediction.py``.


Some useful information
"""""""""""""""""""""""

1. The *RBF* is a real-valued function whose value depends only on the distance from the origin, so that: :math:`\phi(x)=\phi(||x||)`.

2. The *KGM* is a statistical prediction of a function at *untried inputs*. KGM is a flexible and robust technique to build fast *surrogate models* based on small experimental designs.


Block 5 - UQ
^^^^^^^^^^^^

UQ (or *Uncertainty Quantification*) is used as a method to evaluate the results.

.. code-block:: python

    "uq": {
        "sample": 1000,
        "pdf": ["Uniform(1., 3.1415)", "Uniform(1., 3.1415)"],
        "type": "aggregated",
        "method": "sobol"
    }

+ ``method``: type of Sobol analysis: *sobol*, *FAST* (or *Fourier Amplitude Sensitivity Testing*) (if FAST, no second-order indices).
+ ``type``: type of indices we want: *aggregated* or *block*.
+ ``sample``: use a test method: *Ishigami*.
+ ``pdf`` *Probability density function* for uncertainty propagation. Enter the PDF of the inputs: x1: Normal(mu, sigma), x2: Uniform(inf, sup).



