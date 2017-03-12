.. _introduction:

BATMAN introduction
===================

A surrogate tool
----------------

The use of *Computational Fluid Dynamics* (CFD) has proven to be reliable, faster and cheaper than experimental campaigns in an industrial context. However, sensitivity analysis needs a large amount of simulation which is not feasible when using complex codes that are time and resources consuming. This is even more true in *LES* context as we are trying to have a representative simulation. The only solution to overcome this issue is to construct a model that would estimate a given QoI in a given range. This model requires a realistic amount of evaluation of the detail code. The general procedure to construct it consists of:

* Generate a sample space:
    Generate a set of data from which to run the code. A solution is called a *snapshot*.

* Learn the link between the input the output data:
    From the previously generated set of data, we can compute a model also called a response surface. A model is build using gaussian process [Rasmussen2006]_ or polynomial chaos expansion [Najm2009]_.

* Predict a solution from a new set of input data:
    The model can finaly be used to interpolate a new snapshot from a new set of input data.

.. image:: ./fig/surrogate.pdf

.. warning:: The model cannot be used for extrapolation. Indeed, it has been constructed using a sampling of the space of parameters. If we want to predict a point which is not contained within this space, the error is not contained as the point is not balanced by points surrounding it. As a famous catastrophe, an extrapolation of the physical properties of an o-ring of the *Challenger* space shuttle lead to an explosion during lift-off [Draper1995]_.

Once this model has been constructed, using *Monte Carlo* sampling we can compute Sobol' indices, etc. Indeed, this model is said to be costless to evaluate, this is why the use of the *Monte Carlo* sampling is feasible. To increase convergence, we can still use the same methods as for the DOE.

Both *Proper Orthogonal Decomposition* (POD) and *Kriging* (*PC*, *RBF*, etc.) are techniques that can interpolate data using snapshots. The main difference being that POD compresses the data it uses to use only the relevant modes whereas Kriging method doesn't reduce the size of the used snapshots. On the other hand, POD cannot reconstruct data from a domain missing ones [Gunes2006]_. Thus, the strategy used by BATMAN consists in:

0. Create a Design of Experiments,
1. Optionaly use POD reconstruction in order to compress data,
2. Construct a surrogate model [on POD's coefficients],
3. Interpolate new data.


.. seealso:: More details about :ref:`space`, :ref:`pod` or :ref:`surrogate`.


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

* :py:mod:`batman.ui`: command line interface,
* :py:mod:`batman.driver`: contains the main functions,
* :py:mod:`batman.uq`: uncertainty quantification,
* :py:mod:`batman.surrogate`: constructs the surrogate model,
* :py:mod:`batman.space`: defines the (re)sampling space,
* :py:mod:`batman.pod`: constructs the POD,
* :py:mod:`batman.tasks`: defines the context to compute each snapshot from,
* :py:mod:`batman.functions`: defines usefull test functions,
* :py:mod:`batman.misc`: defines the logging configuration and the settings schema.

Using it
........

After BATMAN has been installed, ``batman`` is available as a command line tool or it can be imported in python. The CLI is defined in :py:mod:`batman.ui`. The module imports the package and use the function defined in :py:mod:`batman.driver`.

Thus BATMAN is launched using::

    batman settings.json

.. seealso:: The definition of the case is to be filled in ``settings.json``. Refer to :ref:`settings`.

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
* ``RAE2822`` is a *2-input_parameters* that launches an *elsA* case,
* ``Flamme_1D`` is a *2-input_parameters* that launches an *AVBP* case.

In every case, there is ``README.rst`` file that summarize and explain it.

References
----------

.. [Rasmussen2006] CE. Rasmussen and C. Williams: Gaussian processes for machine learning. MIT Press. 2006. ISBN: 026218253X
.. [Najm2009] H. N. Najm, Uncertainty Quantification and Polynomial Chaos Techniques in Computational Fluid Dynamics, Annual Review of Fluid Mechanics 41 (1) (2009) 35–52. DOI:10.1146/annurev.fluid.010908.165248.
.. [Gunes2006] H. Gunes, S. Sirisup and GE. Karniadakis: “Gappydata:ToKrigornottoKrig?”. Journal of Com putational Physics. 2006. DOI:10.1016/j.jcp.2005.06.023
.. [Draper1995] D. Draper: “Assessmentand Propagation ofModelUncertainty”. Journal of the Royal Statistical Society. 1995.

