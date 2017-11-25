.. _technical_doc:

Technical Documentation
=======================

The use of computational simulations in many areas of science has proven to be reliable, faster and cheaper than experimental campaigns. However, the parametric analysis needs a large amount of simulations which is not feasible when using huge codes that are time and resources consuming. An efficient solution to overcome this issue is to construct models that are able to estimate correctly the responds of the codes. These models, called *Surroagte Models*, require a realistic amount of evaluation of the codes and the general procedure to construct them consists in:

* Generating a sample space:
    Produce a set of data from which to run the code. The points contained in this set all called  *snapshot*.

* Learning the link between the input the output data:
    From the previously generated set of data, we can compute a model, which  is build using gaussian process [Rasmussen2006]_ or polynomial chaos expansion [Najm2009]_.

* Predictng solutions from a new set of input data:
    The model can finaly be used to interpolate a new snapshot from a new set of input data.

.. image:: ./fig/surrogate.pdf

.. warning:: The model cannot be used for extrapolation. Indeed, it has been constructed using a sampling of the space of parameters. If we want to predict a point which is not contained within this space, the error is not contained as the point is not balanced by points surrounding it. As a famous catastrophe, an extrapolation of the physical properties of an o-ring of the *Challenger* space shuttle lead to an explosion during lift-off [Draper1995]_.


Both *Proper Orthogonal Decomposition* (POD) and *Kriging* (*PC*, *RBF*, etc.) are techniques that can interpolate data using snapshots. The main difference being that POD compresses the data it uses to use only the relevant modes whereas Kriging method doesn't reduce the size of the used snapshots. On the other hand, POD cannot reconstruct data from a domain missing ones [Gunes2006]_. Thus, the strategy used by BATMAN consists in:

.. toctree::
   :maxdepth: 1
   :numbered:

   Create a Design of Experiments <technical_documentation/space>
   Optionaly use POD reconstruction in order to compress data <technical_documentation/pod>
   Construct a surrogate model [on POD's coefficients] <technical_documentation/surrogate>
   Do statistical analysis <technical_documentation/uq>
   Visualize the results <technical_documentation/visualization>


As a reference, here is some bibliography: 

.. toctree::
   :maxdepth: 1

   technical_documentation/bibliography

Content of the package
----------------------

The BATMAN package includes: 

* ``doc`` contains the documentation,
* ``batman`` contains the module implementation,
* ``test_cases`` contains some example.

General functionment
....................

The package is composed of several python modules which are self contained within the directory ``batman``.
Following is a quick reference:

* :py:mod:`ui`: command line interface,
* :mod:`space`: defines the (re)sampling space,
* :py:mod:`surrogate`: constructs the surrogate model,
* :py:mod:`uq`: uncertainty quantification,
* :mod:`visualization`: uncertainty visualization,
* :py:mod:`pod`: constructs the POD,
* :py:mod:`driver`: contains the main functions,
* :py:mod:`tasks`: defines the context to compute each snapshot from,
* :py:mod:`functions`: defines usefull test functions,
* :py:mod:`misc`: defines the logging configuration and the settings schema.

Using it
........

After BATMAN has been installed, ``batman`` is available as a command line tool or it can be imported in python. The CLI is defined in :py:mod:`ui`. The module imports the package and use the function defined in :py:mod:`driver`.

Thus BATMAN is launched using::

    batman settings.json

An ``output`` directory is created and it contains the results of the computation splited across the following folders: 

* ``snapshots``,
* ``surrogate``,
* [``predictions``],
* [``uq``].

Content of ``test_cases``
.........................

This folder contains ready to launch examples: 

* ``Basic_function`` is a simple *1-input_parameter* function,
* ``Michalewicz`` is a *2-input_parameters* non-linear function,
* ``Ishigami`` is a *3-input_parameters*,
* ``G_Function`` is a *4-input_parameters*,
* ``Channel_Flow`` is a *2-input_parameters* with a functionnal output,
* ``Mascaret`` make use of MASCARET open source software (not included).

In every case folder, there is ``README.rst`` file that summarizes and explains it.
