.. _introduction:

JPOD introduction
=================

A surrogate tool
----------------

The use of *Computational Fluid Dynamics* (CFD) has proven to be reliable, faster and cheaper than experimental campaigns in an industrial context. However, sensitivity analysis needs a large amount of simulation which is not feasible when using complex codes that are time and resources consuming. This is even more true in *LES* context as we are trying to have a representative simulation. The only solution to overcome this issue is to construct a model that would estimate a given QoI in a given range. This model requires a realistic amount of evaluation of the detail code. The general procedure to construct it consists of:

* Generate a sample space:
    Generate a set of data from which to run the code. A solution is called a *snapshot*.

* Learn the link between the input the output data:
    From the previously generated set of data, we can compute a model also called a response surface. A model is build using gaussian process [Rasmussen2006]_.

* Predict a solution from a new set of input data:
    The model can finaly be used to interpolate a new snapshot from a new set of input data.

.. image:: ./fig/surrogate.pdf

.. warning:: The model cannot be used for extrapolation. Indeed, it has been constructed using a sampling of the space of parameters. If we want to predict a point which is not contained within this space, the error is not contained as the point is not balanced by points surrounding it. As a famous catastrophe, an extrapolation of the physical properties of an o-ring of the *Challenger* space shuttle lead to an explosion during lift-off [Draper1995]_.

Once this model has been constructed, using *Monte Carlo* sampling we can compute Sobol' indices, etc. Indeed, this model is said to be costless to evaluate, this is why the use of the *Monte Carlo* sampling is feasible. To increase convergence, we can still use the same methods as for the DOE.

Both *Proper Orthogonal Decomposition* (POD) and *Kriging* are techniques that can interpolate data using snapshots. The main difference being that POD compresses the data it uses to use only the relevant modes whereas Kriging method doesn't reduce the size of the used snapshots. On the other hand, POD cannot reconstruct data from a domain missing ones [Gunes2006]_. Thus, the strategy used by JPOD consists in:

0. Create a Design Of Experiments,
1. Use POD reconstruction in order to compress data,
2. Use Kriging interpolation on POD's coefficients,
3. Interpolate missing data.


.. seealso:: More details about :ref:`space`, :ref:`pod` or :ref:`surrogate`.


Content of the package
----------------------

The JPOD package includes 2 repository:

* ``kernel`` contains the JPOD package and its implementation,
* ``test_cases`` contains some example.


General functionment
....................

The package is composed of several python modules which are self contained within the directory ``kernel/jpod``.
Following is a quick reference:

* :py:mod:`ui`: command line interface,
* :py:mod:`driver`: contains the main functions,
* :py:mod:`uq`: uncertainty quantification,
* :py:mod:`surrogate`: constructs the surrogate model,
* :py:mod:`space`: defines the (re)sampling space,
* :py:mod:`pod`: constructs the POD,
* :py:mod:`tasks`: defines the context to compute each snapshot from,
* :py:mod:`misc`: defines the logging configuration.

After JPOD has been installed, ``jpod`` is available as a command and it can be imported in python. 
It is a link to :py:mod:`ui`. The module imports the package and use the function defined in :py:mod:`driver`.

Thus JPOD is launched using::

    python jpod task.py

An ``output`` directory is created and it contains the results of the computations of all the *snapshots*, the *pod* and the *predictions*.


.. image:: ./fig/UML.png


Content of ``test_cases``
.........................



.. [Thie2009] T. Braconnier and M. Ferrier: Jack Proper Orthogonal Decomposition (JPOD) for Steady Aerodynamic Model. Tech. rep. 2009
.. [Rasmussen2006] CE. Rasmussen and C. Williams: Gaussian processes for machine learning. MIT Press. 2006. ISBN: 026218253X
.. [Damblin2013] G. Damblin, M. Couplet, B. Iooss: Numerical studies of space filling designs : optimization of Latin Hypercube Samples and subprojection properties. Journal of Simulation. 2013.
.. [Gunes2006] H. Gunes, S. Sirisup and GE. Karniadakis: “Gappydata:ToKrigornottoKrig?”. Journal of Com putational Physics. 2006. DOI: 10. 1016/j.jcp.2005.06.023

.. [Draper1995] D. Draper: “Assessmentand Propagation ofModelUncertainty”. Journal of the Royal Statistical Society. 1995.

