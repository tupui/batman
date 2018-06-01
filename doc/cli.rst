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
            "method": "halton",
            "distributions": ["Uniform(15., 60.)", "BetaMuSigma(4035, 400, 2500, 6000).getDistribution()"],
            "discrete": 0
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
      ``["Uniform(15., 60.)", "Normal(4035., 400.)"]``,
    * [``discrete``]: index of the parameter which is discrete.

+ [``resampling``]: to do resampling, fill this dictionary

    * ``resamp_size``: number of point to add in the parameter space.
    * ``method``: to be choosen from ``sigma``, ``loo_sigma``, ``loo_sobol``, ``hybrid``, ``discrepancy``, ``optimization``, ``extrema``.
    * [``delta_space``]: the percentage of space to shrink to not resample close to boundaries. For ``0.08``,
      the available space for resampling will be shrinked by 8%.
    * [``hybrid``]: if method is ``hybrid``. You have to define a generator which is a list
      ``[["method", n_snapshot]]``.
    * [``extrema``]: to be used with ``optimization``, will find the global maximum if set to ``max``.
    * [``q2_criteria``]: stopping criterion based on the quality estimation of the model.

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
        "plabels": ["x1", "x2"],
        "flabels": ["X", "F"],
        "psizes": [1, 1],
        "fsizes": [2, 5],
        "io": {
            "space_fname": "sample-space.json",
            "space_format": "json",
            "data_fname": "sample-data.json",
            "data_format": "json"
        },
        "provider": ...  # comes in 3 flavors
    }

+ ``max_workers``: maximum number of simultaneous running snapshots.
+ ``plabels``: names of the parameters that serve as coordinates of a snapshot point.
+ ``flabels``: names of the variables to treat that are contained in a snapshot.
+ [``psizes``]: number of components of each parameter.
+ [``fsizes``]: number of components of each variable.
+ [``io``]: change default values for the global input/output files.
    * [``space_fname``]: basename for files storing the point coordinates ``plabels``. 
    * [``space_format``]: ``json`` (default), ``csv``, ``npy``, ``npz``.
    * [``data_fname``]: basename for files storing values associated to ``flabels``.
    * [``data_format``]: ``json`` (default), ``csv``, ``npy``, ``npz``.

The ``provider`` block defines what a simulation is. It comes in two flavors.
A simulation can either be the result of a user-provided python function,
or it can be an external program that produces a data file.

Provider Function - User-provided python function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot data is produced by calling a python function.
No I/O is performed by default, it is the provider that shall bring the best performance.

.. code-block:: python

    "provider": {
        "type": "function",
        "module": "my.python.module",
        "function": "f",
        "discover": "some/*/snapshot/directories"
    }

+ ``type``: type of provider. Must be set to ``function``.
+ ``module``: name of the python module to load.
+ ``function``: name of the function in ``module``. Called whenever a snapshot is required.
+ [``discover``]: UNIX-style pattern matching path to directories carrying snapshot files.
  File names and formats are the ones set in ``io`` block.

Provider File - Read data from files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot data is read from files.
An erro is raised if a snpashot is request for a point not covered by user's files.

.. code-block:: python

    "provider": {
        "type": "file",
        "file_pairs": [['path-to/space-file.json', 'path-to/data-file.csv'],
                       ['toto/space.json', 'tata/data.csv']],
        "discover": "some/*/snapshot/directories"
    }

+ ``type``: type of provider. Must be set to ``file``.
+ ``file_pairs``: list of couples of files. 1st file contains space, 2nd one contains data.
  File name are absolute or relative paths to the file.
  File formats are the ones set in ``io`` block.
+ [``discover``]: UNIX-style pattern matching path to directories carrying snapshot files.
  File names and formats are the ones set in ``io`` block.

Provider Job - Coupling with 3rd-party program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot data is produced by running a 3rd-party program.
The program is given as a shell command line to be executed in a context directory.
Coupling between BATMAN and external program is done through files.

In case of expensive program, the snapshots can be send to an external host.

.. code-block:: python

    "provider": {
    "type": "job",
    "command": "bash script.sh",
    "context_directory": "data",
    "coupling": {
        "coupling_directory": "batman-coupling",
        "input_fname": "sample-space.npy",
        "input_format": "npy",
        "output_fname": "sample-data.npz",
        "output_format": "npz"
    }
    "clean": false,
    "discover": "some/*/snapshot/directories",
    "hosts": [
            {
                "hostname": "nemo",
                "remote_root": "TOTO",
                "username": "batman",
                "password": "Iron man sucks!",
                "weight": 0.2
            },
            {
                "hostname": "occigen",
                "remote_root": "TATA",
                "username": "batman",
                "password": "Iron man sucks!",
                "weight": 0.8
            }
        ]
    }

+ ``type``: type of provider. Must be set to ``job``.
+ ``command``: command to run the external program. Launched from ``context_directory``. The program shall read its input parameters from ``coupling_directory/point_filename`` and write its outputs to ``coupling_directory/data_filename``. File names and formats are the one set in ``io`` block.
+ ``context_directory``: directory containing input data and script for building snapshot data files.
  .. note:: BATMAN always keeps ``context_directory`` untouched. Actual workdirs are copies with symlinks to directory content.
+ [``coupling``]:
    * [``coupling_directory``]: directory in ``context_directory`` that will contain input parameters and output file. Its creation and deletion is handled by BATMAN.
    * [``input_fname``]: basename for files storing the point coordinates ``plabels``.
    * [``input_format``]: ``json`` (default), ``csv``, ``npy``, ``npz``.
    * [``output_fname``]: basename for files storing values associated to ``flabels``.
    * [``output_format``]: ``json`` (default), ``csv``, ``npy``, ``npz``.

+ [``clean``]: delete after run working directories.
+ [``discover``]: UNIX-style pattern matching path to directories carrying snapshot files.
+ [``hosts``]: list of different remote hosts to connect to with the following options:
    * ``hostname``: Remote host to connect to.
    * ``remote_root``: Remote folder to create and store data in.
    * [``username``]: username.
    * [``password``]: password.
    * [``weight``]: load balancing between hosts. Can use any units. Ex. with two hosts: 0.2, 0.8 or 20, 80 are equivalent.

  This functionality is based on *ssh* and *sftp*. So user configuration in ``~/.ssh/config`` is used by default.
  Also, private keys are used if located in default folder.

Optionnal Block 3 - Surrogate
-----------------------------

Set up the surrogate model strategy to use. See :ref:`Surrogate <surrogate>`.

.. code-block:: python

    "prediction": {
        "method": "kriging",
        "predictions": [[30, 4000], [35, 3550]]
    }

+ ``method``: method used to generate a snapshot one of *rbf* (Radial Basic Function), *kriging*, *pc* (polynomial chaos expension) or *evofusion* method.
  Otherwise it can be a string that define a model from Scikit-Learn regressors. Ex ``"RandomForestRegressor()"``
+ [``predictions``]: set of points to predict.

For *kriging* the following extra attributes **can** be set: 

+ [``kernel``]: kernel to use. Ex: ``"ConstantKernel() + Matern(length_scale=1., nu=1.5)"``.
+ [``noise``]: noise level as boolean or as a float.
+ [``global_optimizer``]: whether to do global optimization or gradient based optimization to estimate hyperparameters.

For *pc* the following extra attributes **must** be set: 

+ ``strategy``: either using quadrature, standard least square, or Sparse Cleaning Strategy:  *Quad* , *LS*, *SparseLS*.
+ ``degree``: the polynomial degree. If *SparseLS* is selected as ``strategy``, this is not used actually.

For *pc* the following extra attributes **can** be set: 

+ [``sparse_param``]: a dictionary containing either parameters useful to Sparse Cleaning Strategy (if *SparseLS* is chosen),  or the parameter for truncating the basis in an hyperbolic fashion. 

.. code-block:: python

    "sparse_param": {
        "max_considered_terms": 400},
        "most_significant":  30},
        "significance_factor": 1e-3},
        "hyper_factor": 0.5}
    }

    - ``max_considered_terms``: (int) The maximun number of terms used for the trials by the Sparse Cleaning Technique before giving the best solution,
    - ``most_significant``: (int) The maximum dimension of the basis that the Sparse Cleaning Techniques gives as an output,
    - ``significance_factor``: (float) The threshold value below which the basis member is discarded,
    - ``hyper_factor``: (float) The value for hyperbolic truncation. Value to be in range (0,1], where  1.0 = Linear Truncation. The lower the value, the sparser the base.

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

+ [``test``]: use a test method for indices comparison and quality calculation. Use one of: *Rosenbrock*, *Michalewicz*, *Ishigami*, *G_Function*, *Channel_Flow*,
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
        "kiviat_fill": true,
        "2D_mesh": {
                 "fname": "mesh_file.csv",
                 "format": "csv",
                 "xlabel": "x label",
                 "ylabel": "y label",
                 "flabel": ["Variable of interest"],
                 "vmins" = [0.1]
        }
     }

+ [``bounds``]: Floats. Response surface boundaries. Those boundaries should be included inside the space corners defined in the Space of Parameters block. Default values are the space corners,
+ [``doe``]: Boolean. If *true*, the Design of Experiment is represented on the response surface by black dots. Default value is *false*,
+ [``resampling``]: Boolean. If *true*, Design of Experiment corresponding to the resampling points are displayed in a different color. Such points are represented by red triangles. Only activates if doe is *true*,
+ [``axis_disc``]: Integers. Discretisation of each axis. Indicated value for the x and the y axis modify the surface resolution, while values corresponding the the 3rd and 4th parameters impact the frame number per movie and the movie number,
+ [``flabel``]: String. Name of the cost function,
+ [``xlabel``]: String. Name of the abscissa,
+ [``ylabel``]: String. Name of the ordinate,
+ [``plabels``]: Strings. Name of the input parameters to be plotted on each axis,
+ [``feat_order``]: Integers. Associate each input parameter to an axis, the first indicated number corresponding to the parameter to be plotted on the x-axis, etc... A size equal to the input parameter number is expected, all integers from 1 to the parameter number should be used. Default is *[1, 2, 3, 4]*,
+ [``ticks_nbr``]: Integer. Number of ticks on the colorbar (Display n-1 colors). Default is *10*,
+ [``range_cbar``]: Floats. Minimum and maximum values on the colorbar,
+ [``contours``]: Floats. Values of the iso-contours to be plotted on the response surface,
+ [``kiviat_fill``]: Boolean. If *true*, will fill the surface of the Kiviat plot,
+ [``2D_mesh``]: Block containing all options related to representation of statistical variable of interest on a 2D mesh provided by the user. Possible options are:
        + ``fname``: String. Name of the input file containing the mesh coordinates,
        + ``format``: String. Format of the input file,
        + ``xlabel``: String. Name of the abscissa,
        + ``ylabel``: String. Name of the ordinates,
        + ``flabels``: List(String). Names of the variables of interest,
        + ``vmins``: List(Float). Minimum values of the variables of interest to be plotted for data filtering.


.. py:module:: driver
.. py:currentmodule:: driver

Driver module
-------------

.. automodule:: batman.driver
   :members:
   :undoc-members:
