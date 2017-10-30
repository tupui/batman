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
      *sobol*, *sobolscramble*, *lhs* (Latin Hypercube Sampling), *lhsc* (Latin Hypercube  Sampling Centered) or *lhsopt* (optimized LHS).
      Another possibilitie is to set a list of distributions. Ex for two input variables: ``["Uniform(15., 60.)", "Normal(4035., 400.)"]``

+ [``resampling``]: to do resampling, fill this dictionary

    * ``delta_space``: the percentage of space to shrink to not resample close to boundaries. For ``0.08``,
      the available space for resampling will be shrinked by 8%.
    * ``resamp_size``: number of point to add in the parameter space.
    * ``method``: to be choosen from ``sigma``, ``loo_sigma``, ``loo_sobol``, ``hybrid``, ``discrepancy``, ``optimization``, ``extrema``.
    * [``hybrid``]: if method is ``hybrid``. You have to define a generator which is a list
      ``[["method", n_snapshot]]``
    * ``q2_criteria``: stopping criterion based on the quality estimation of the model.

The method used to create the DoE is paramount. It ensures that that the physics will be captured correclty all over the domain of interest, see :ref:`Space <space>`. All *faure*, *halton* and *sobol* methods are low discrepancy sequences with good filling properties.

Regarding the resampling, all methods need a good initial sample. Meanning that the quality is about :math:`Q_2\sim0.5`. ``loo_sigma, loo_sobol`` work better than ``sigma`` in high dimentionnal cases (>2).

.. warning:: If using a PC surrogate model, the only possibilities are ``discrepancy`` and ``extrema``. Furthermore, sampling ``method`` must be set as a list of distributions.

Block 2 - Snapshot provider
---------------------------

A snapshot defines a simulation.

.. code-block:: python

    "snapshot": {
        "max_workers": 5,
        "io": {
            "shapes": {
                "0": [
                    [400]
                ]
            },
            "format": "fmt_tp_fortran",
            "variables": ["X", "F"],
            "point_filename": "header.py",
            "filenames": {
                "0": ["function.dat"]
            },
            "template_directory": "output/snapshots/0/batman-data/header.py",
            "parameter_names": ["x1", "x2"]
        },
        "provider": {
            "command": "bash",
            "timeout": 20,
            "context": "data",
            "script": "data/script.sh",
            "clean": false,
            "private-directory": "batman-data",
            "data-directory": "cfd-output-data",
            "restart": "False"
        }

+ ``max_workers``: maximum number of simultaneous running snapshot,
+ ``shapes``: shape per variable and per file,
+ ``format``:  ``npz``, ``fmt_tp_fortran`` (BATMAN) or all Antares formats if installed,
+ ``variables``: names of the variables to treat and contained in a snapshot,
+ ``point_filename``: name of the file that contains the coordinates of a point in the parameter space,
+ ``filenames``: dictionary of list, if several filenames, several independant analysis are done. It contains the output ``variables`` of a snapshot to use by BATMAN,
+ ``template_directory``: path to the point file name of the first snapshot, used to restart from existing jobs not created with BATMAN,
+ ``parameter_names``: names of the parameters.

The ``provider`` defines what is a simulation. If we simply want to evaluate a python function, we can pass the ``function_name`` as a string. ``function_name.py`` will be imported and the function named ``f`` will be used. Otherwize, for a complexe case, use a dictionary:

+ ``command``: command to use to launch the script,
+ ``timeout``: timeout of each jobs in seconds,
+ ``context``: directiory where the data for the BATMAN computations are stored,
+ ``script``: path of the script or batch to run,
+ ``clean``: delete after run all except what is inside ``private-directory``
+ ``private-directory``: folder containing the ``point_filename`` and the result of the snapshot
+ ``data-directory``: output folder to store the ``filenames``,
+ ``restart``: restart the computation if the job has failed.

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
+ ``method``: type of Sobol analysis: *sobol*, *FAST* (Fourier Amplitude Sensitivity Testing) (if FAST, no second-order indices).
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


.. py:module:: driver
.. py:currentmodule:: driver

Driver module
-------------

.. automodule:: batman.driver
   :members:
   :undoc-members:
