.. _tutorial:


Tutorial
========


Introduction
------------


| After installing ``JPOD`` tool, you can create the working directory anywhere on your PC. In fact, you can find the examples in the ``test-cases`` subdirectory of ``JPOD`` installer directory (see the directory tree below).

.. code::

     JPOD
     |
     ├── kernel
     |   |
     |   ├── jpod
     |       |
     |       ├── ui.py
     |
     ├── test-cases
         |
         ├── Michalewicz
         |   |
         |   ├── data
         |   |   |
         |   |   ├── function.py
         |   |
         |   ├── scripts
         |       |
         |       ├── settings_template.py
         |       |
         |       ├── task.py       
         |
         ├── RAE2822
         |   |
         |   ├── data
         |   |
         |   ├── scripts
         |
         ├── etc.

| Generally, each working directory consists of two main parts:

+ Directory ``data``: this directory contains files describing the code of *CFD calculation* (AVBP, elsA, etc.) or the code of *optimization test functions*, for example.

+ Directory ``scripts``: this directory contains python script files which filled by users for setting the JPOD configuration.

|

JPOD step-by-step
-----------------


Step 1: Building directory ``data``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Step 1a: Selecting test configuration
"""""""""""""""""""""""""""""""""""""


| We chose the `Michalewicz function <http://www.sfu.ca/~ssurjano/michal.html>`_ - a multimodal d-dimensional function which has :math:`d!` local minima - for this *test-case*: 

| :math:`f(x)=-\sum_{i=1}^d sin(x_i)sin^{2m}\left(\frac{ix_i^2}{\pi}\right)`

| where *m*: steepness of the valleys and ridges.


.. note:: + It is so difficult to search a global minimum when :math:`m` reaches large value. Therefore, it is recommended that his value is :math:`m = 10`.
          + The function's form is two-dimensional, i.e., :math:`d = 2`. 


| In summary, we have the Michalewicz 2D function as follows:

| :math:`f(x)=-sin(x_1)sin^20\left(\frac{x_1^2}{\pi}\right)-sin(x_2)sin^20\left(\frac{2x_2^2}{\pi}\right)`

|


Step 1b: Creating script file 
"""""""""""""""""""""""""""""


| After selecting the test-case, a script file must be created. So we take the ``function.py`` in the directory ``data`` as an example to describe the information of *optimization test function*.

.. code-block:: python

    F = -1.0-math.sin(X1)*(math.pow(math.sin(X1*X1/math.pi),20.))-math.sin(X2)*(math.pow(math.sin(2*X2*X2/math.pi),20.))

Read more
*********

| For other *optimization functions*, read more in the `website of Derek Bingham <http://www.sfu.ca/~ssurjano/optimization.html>`_ (please contact the author via email: dbingham@stat.sfu.ca).

|

Step 2: Building directory ``scripts``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


| In fact, ``settings_template.py`` which is the most important file in the directory ``scripts`` is built for the test-case *Michalewicz function* . In this file, there are five blocks with different functions:

|

Block 1 - Parameters space
""""""""""""""""""""""""""


| In this block, we create the coordinate with the sampling points and chose the method to generate these points.

.. code-block:: python

    space = {'corners'     : ((1., 1.), (3.1415, 3.1415),),
              'delta_space' : 0.01,                         
              'size_max'    : 21,
              'provider'    : {'method' : 'halton',
                               'size'   : 20,
                              }
            }


Block 2 - Snapshot provider
"""""""""""""""""""""""""""


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


Block 3 - POD
"""""""""""""


POD (or Proper Orthogonal Decomposition) is a approach to help reduce amount of data.

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


Block 4 - Prediction
""""""""""""""""""""

We can built the prediction function with different methods.

.. code-block:: python

    prediction = {'method' : 'kriging',
                  'points' : [ ],
                 }

.. note:: We can fill *directly* the number of points into the brackets or *indirectly* via the script.


Block 5 - UQ
""""""""""""


UQ (or *Uncertainty Quantification*) is used as a method to evaluate the results.

.. code-block:: python

    uq = {'method' : 'sobol',
          'type'   : 'aggregated',
          'sample' : 5000 ,
          'pdf'    : ['Uniform(-2.048, 2.048)',
                      'Uniform(-2.048, 2.048)']
         }


Read more
*********

.. seealso:: find some more information in :ref:`settings` file.

.. note:: + Similarly, you change **only** the *function formula* in the script file and *coordinate* for other optimization test functions.
          + Another way, you can create the script files for the *CFD calculation* cases.

|

Step 3: Running JPOD
^^^^^^^^^^^^^^^^^^^^


| It is executed when we run two python files: ``ui.py`` and ``task.py`` (see the directory tree below).

.. code::

     JPOD
     |
     ├── kernel
     |   |
     |   ├── jpod
     |       |
     |       ├── ui.py
     |
     ├── test-cases
         |
         ├── Michalewicz
         |   |
         |   ├── data
         |   |   |
         |   |   ├── function.py
         |   |
         |   ├── scripts
         |       |
         |       ├── settings_template.py
         |       |
         |       ├── task.py       
         |
         ├── RAE2822
         |   |
         |   ├── data
         |   |
         |   ├── scripts
         |
         ├── etc.

| Finally, you receive the result of JPOD calculation in the ``JPOD.log`` file: 

.. code-block:: bash

    JPOD main ::
        POD summary:
        modes filtering tolerance    : 0.99
        dimension of parameter space : 2
        number of snapshots          : 20
        number of data per snapshot  : 1
        maximum number of modes      : 100
        number of modes              : 1
        modes                        : [ 1.69972346]

|

Step 4: Post-treatment
^^^^^^^^^^^^^^^^^^^^^^


| All of result files located in 3 directories: ``pod``, ``predictions`` and ``snapshots`` of the directory ``output`` (see the directory tree below).

.. code::

     JPOD
     |
     ├── kernel
     |
     ├── test-cases
     |
     ├── output
         |
         ├── pod
         |
         ├── predictions
         |
         ├── snapshots

| You can use these files for post-treatment with some available softwares such as: *paraview*, *tecplot*, etc.

| In this example, here are the images that we obtain with a visual data analysis tool *Tecplot 360*:

|

.. image:: fig/post_2D_1.png

.. image:: fig/post_2D_2.png

|

| A white subfigure at the bottom right describes a *sampling technique*.

| Two subfigures at the bottom left correspond a *Maximum error* :math:`L_max` and a *Coefficient of determination* :math:`R^2` with a trend-line. As close to a linear trend-line, the points get more precision.

| Meanwhile, two subfigures at the top in both cases, from left to right, correspond the *reference* and *prediction functions*. We noticed that the results look quite similar, i.e. the distributions get good solutions.