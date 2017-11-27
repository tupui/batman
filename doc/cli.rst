.. _cli:
.. py:module:: ui

Command Line Interface
======================

Introduction
------------

The file ``settings.py`` contains the configuration of BATMAN. It can be devided into 2 mandatory blocks and 3 optionnal block. Fields in brackets are optionnal and there is no specific order to respect.

.. note:: A prefilled example is shown in ``settings.json`` located in ``test_cases/Snippets``.

Help of the CLI can be triggered with::
    
    batman -h

    usage: BATMAN [-h] [--version] [-v] [-c] [-s] [-o OUTPUT] [-r] [-n] [-u] [-q]
              settings

    BATMAN creates a surrogate model and perform UQ.

    positional arguments:
      settings              path to settings file

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -v, --verbose         set verbosity from WARNING to DEBUG, [default: False]
      -c, --check           check settings, [default: False]
      -s, --save-snapshots  save the snapshots to disk when using a function,
                            [default: False]
      -o OUTPUT, --output OUTPUT
                            path to output directory, [default: ./output]
      -r, --restart         restart pod, [default: False]
      -n, --no-surrogate    do not compute surrogate but read it from disk,
                            [default: False]
      -u, --uq              Uncertainty Quantification study, [default: False].
      -q, --q2              estimate Q2 and find the point with max MSE, [default:
                            False]    

.. note:: Fields in square brackets are optionnals.

Block 1 - Space of Parameters
-----------------------------

First of all, we define the parameter space using an hypercube. Taking the minimal and the maximal value along all coordinates allow to describe it.

.. figure:: fig/hypercube.pdf

   3-dimentionnal hypercube

.. code-block:: python

    "space": {
        "corners": [
            [15.0, 2500.0],
            [60.0, 6000.0]
        ],
        "sampling": {
            "init_size": 4,
            "method": "halton"
        },
        "resampling":{
            "delta_space": 0.08,
            "resamp_size": 0,
            "method": "sigma",
            "hybrid": [["sigma", 4], ["loo_sobol", 2]],
            "q2_criteria": 0.9
        }
    }

+ ``corners``: define the space using the two corners of the hypercube ``[[min], [max]]``,
+ ``sampling``: define the configuration of the sample. This can either be; a list of sample
  as an array_like of shape (n_samples, n_features); or a dictionary with
  the following:

    * ``init_size``: define the initial number of snapshots,
    * ``method``: method to create the DoE, can be *uniform*, *faure*, *halton*,
      *sobol*, *sobolscramble*, *lhs* (Latin Hypercube Sampling), *lhsc* (Latin Hypercube  Sampling Centered) or *lhsopt* (optimized LHS), *saltelli*,
    * [``distributions``]: A list of distributions. Ex for two input variables:
      ``["Uniform(15., 60.)", "Normal(4035., 400.)"]``.

+ [``resampling``]: to do resampling, fill this dictionary

    * ``delta_space``: the percentage of space to shrink to not resample close to boundaries. For ``0.08``,
      the available space for resampling will be shrinked by 8%.
    * ``resamp_size``: number of point to add in the parameter space.
    * ``method``: to be choosen from ``sigma``, ``loo_sigma``, ``loo_sobol``, ``hybrid``, ``discrepancy``, ``optimization``, ``extrema``.
    * [``hybrid``]: if method is ``hybrid``. You have to define a generator which is a list
      ``[["method", n_snapshot]]``
    * ``q2_criteria``: stopping criterion based on the quality estimation of the model.

The method used to create the DoE is paramount. It ensures that that the physics
will be captured correclty all over the domain of interest, see :ref:`Space <space>`.
All *faure*, *halton* and *sobol* methods are low discrepancy sequences with
good filling properties. *saltelli* is particular as it will create a DoE for
the computation of *Sobol'* indices using *Saltelli*'s formulation.

When *distribution* is set, a join distribution is built an is used to perform
an inverse transformation (inverse CDF) on the sample. This allows to have a
low discrepancy sample will still following some distribution.

Regarding the resampling, all methods need a good initial sample. Meanning that the quality is about :math:`Q_2\sim0.5`. ``loo_sigma, loo_sobol`` work better than ``sigma`` in high dimentionnal cases (>2).

.. warning:: If using a PC surrogate model, the only possibilities are ``discrepancy`` and ``extrema``. Furthermore, sampling ``method`` must be set as a list of distributions.


Block 2 - Snapshot specification
--------------------------------

A snapshot defines a simulation.

.. code-block:: python

    "snapshot": {
        "max_workers": 5,
        "parameters": ["x1", "x2"],
        "variables": ["X", "F"],
        "io": {
            "point_filename": "point.json",
            "data_filename": "point.dat",
            "data_format": "fmt_tp_fortran"
        },
        "provider": ...  # comes in 2 flavors
    }

+ ``max_workers``: maximum number of simultaneous running snapshots.
+ ``parameters``: names of the parameters that serve as coordinates of a snapshot point.
+ ``variables``: names of the variables to treat and contained in a snapshot.
+ ``point_filename``: name of the json file that contains the values associated to ``parameters``.
+ ``data_filename``: name of the file that contains the output ``variables`` of a snapshot.
+ ``data_format``: ``npz``, ``fmt_tp_fortran`` (BATMAN) or all Antares formats if installed.

The ``provider`` block defines what a simulation is. It comes in two flavors.
A simulation can either be the result of a user-provided python function,
or it can be an external program that produces a data file.

Provider Plugin - User-provided python function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot data is produced by calling a python function.
No I/O is performed, it is the provider that shall bring the better performance.

.. code-block:: python

    "provider": {
        "type": "plugin",
        "module": "my.python.module",
        "function": "f"
    }

+ ``type``: type of provider. Must be set to ``plugin``.
+ ``module``: name of the python module to load.
+ ``function``: name of the function in ``module``. Called whenever a snapshot is required.

Provider File - Read data from files. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot data is read from files.
If a file for a particular snapshot doesn't exist, it is produce by executing  
a user-specified external program. Snapshots existing in standard directories
are automatically discovered by this provider.

.. code-block:: python

    "provider": {
        "type": "file",
        "discover_from": "my/existing/snapshot/directory",
        "context_directory": "data",
        "coupling_directory": "batman-coupling",
        "command": "bash script.sh",
        "clean": false
    }

+ ``type``: type of provider. Must be set to ``file``.
+ ``discover_from``: path to a directory containing user-provided snapshots. (optional)
+ ``context_directory``: directory containing input data and script for building snapshot data files.
+ ``coupling_directory``: directory in ``context_directory`` that will contain input parameters and output file. Its creation and deletion is handled by BATMAN.
+ ``command``: command to run the external program. Launched from ``context_directory``. The program shall read its input parameters from ``coupling_directory/point_filename`` and write its outputs to ``coupling_directory/data_filename``.
+ ``clean``: delete after run all but snapshot files in execution directory. Content in ``context_directory`` is always preserved.

.. note:: Not specifying ``context_directory`` or ``command`` means no job were specified. It can make sens for those who want to provide their own snapshot data through the ``discover_from`` directory.
.. warning:: BATMAN will crash if it needs a snapshot point that were not provided and no job were specified !


Optionnal Block 3 - Surrogate
-----------------------------

Set up the surrogate model strategy to use. See :ref:`Surrogate <surrogate>`.

.. code-block:: python

    "prediction": {
        "method": "kriging",
        "predictions": [[30, 4000], [35, 3550]]
    }

+ ``method``: method used to generate a snapshot one of *rbf* (Radial Basic Function), *kriging*, *pc* (polynomial chaos expension) or *evofusion* method.
+ [``predictions``]: set of points to predict.

For *kriging* the following extra attributes **can** be set: 

+ [``kernel``]: kernel to use. Ex: ``"ConstantKernel() + Matern(length_scale=1., nu=1.5)"``.
+ [``noise``]: noise level as boolean or as a float.

For *pc* the following extra attributes **must** be set: 

+ ``strategy``: either using quadrature or least square one of *Quad* or *LS*.
+ ``degree``: the polynomial degree.

.. note:: When using *pc*, the ``sampling`` must be set to a list of distributions.

For *evofusion* the following extra attributes **must** be set: 

+ ``cost_ratio``: cost ratio in terms of function evaluation between high and low fidelity models.
+ ``grand_cost``: total cost of the study in terms of number of function evaluation of the high fidelity model.

.. note:: We can fill *directly* the number of points into the brackets or *indirectly* using the script ``prediction.py`` located in ``test_cases/Snippets``.


Optionnal Block 4 - UQ
----------------------

Uncertainty Quantification (UQ), see :ref:`UQ <uq>`.

.. code-block:: python

    "uq": {
        "test": "Channel_Flow"
        "sample": 1000,
        "method": "sobol"
        "pdf": ["Normal(4035., 400)", "Uniform(15, 60)"],
        "type": "aggregated",
    }

+ ``test``: use a test method for indices comparison and quality calculation. Use one of: *Rosenbrock*, *Michalewicz*, *Ishigami*, *G_Function*, *Channel_Flow*,
+ ``sample``: number of points per sample to use for SA,
+ ``method``: type of Sobol analysis: *sobol*, *FAST* (Fourier Amplitude Sensitivity Testing). If FAST, no second-order indices are computed and defining a surrogate model is mandatory.
+ ``type``: type of indices: *aggregated* or *block*.

+ ``pdf`` *Probability density function* for uncertainty propagation. Enter the PDF of the inputs,
  as list of openturns distributions. Ex: x1-Normal(mu, sigma), x2-Uniform(inf, sup)
  => ``["Uniform(15., 60.)", "Normal(4035., 400.)"]``


Optionnal Block 5 - POD
-----------------------

POD (or Proper Orthogonal Decomposition) is a approach to help reduce amount of data.

.. code-block:: python

     "pod": {
        "dim_max": 100,
        "tolerance": 0.99,
        "type": "static"
     }

+ ``tolerance``: tolerance of the modes to be kept. A percentage of the sum of the singular values, values that account for less than this tolerance are ignored,
+ ``dim_max``: maximum number of modes to be kept,
+ ``type``: type of POD to perform: *static* or *dynamic*.

The dynamic POD allows to update the POD once a snapshot is availlable. Hence a POD can be restarted when doing resampling for example.


Optionnal Block 6 - Visualization
---------------------------------

Set up for the visualization options. Batman creates a response function (1 input parameter), response surfaces (2 to 4 input parameters) or a Kiviat graph (more than 4 input parameters). See :ref:`Visualization <visualization>`.

.. code-block:: python

     "visualization": {
        "bounds": [
            [15.0, 2500.0],
            [60.0, 6000.0]
        ],
        "doe": true,
        "resampling": true,
        "axis_disc": [20, 20],
        "flabel": "Cost function",
        "plabels": ["X", "Y"],
        "feat_order": [1, 2],
        "ticks_nbr": 14,
        "range_cbar": [0.0, 2.3],
        "contours": [0.5, 1.0, 1.5],
        "kiviat_fill": true
     }

+ ``bounds``: Floats. Response surface boundaries. Those boundaries should be included inside the space corners defined in the Space of Parameters block. Default values are the space corners,
+ ``doe``: Boolean. If *true*, the Design of Experiment is represented on the response surface by black dots. Default value is *false*,
+ ``resampling``: Boolean. If *true*, Design of Experiment corresponding to the resampling points are displayed in a different color. Such points are represented by red triangles. Only activates if doe is *true*,
+ ``axis_disc``: Integers. Discretisation of each axis. Indicated value for the x and the y axis modify the surface resolution, while values corresponding the the 3rd and 4th parameters impact the frame number per movie and the movie number,
+ ``flabel``: String. Name of the cost function,
+ ``xlabel``: String. Name of the abscissa,
+ ``plabels``: Strings. Name of the input parameters to be plotted on each axis,
+ ``feat_order``: Integers. Associate each input parameter to an axis, the first indicated number corresponding to the parameter to be plotted on the x-axis, etc... A size equal to the input parameter number is expected, all integers from 1 to the parameter number should be used. Default is *[1, 2, 3, 4]*,
+ ``ticks_nbr``: Integer. Number of ticks on the colorbar (Display n-1 colors). Default is *10*,
+ ``range_cbar``: Floats. Minimum and maximum values on the colorbar,
+ ``contours``: Floats. Values of the iso-contours to be plotted on the response surface,
+ ``kiviat_fill``: Boolean. If *true*, will fill the surface of the Kiviat plot.


.. py:module:: driver
.. py:currentmodule:: driver

Driver module
-------------

.. automodule:: batman.driver
   :members:
   :undoc-members:
