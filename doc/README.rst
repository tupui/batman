.. _readme:

README
======

Introduction
------------

The use of computational fluid dynamic (CDF) technics has proven to be reliable, faster and cheaper than experimental measure. However, sensitivity analysis needs a large amount of simulation which is not feasible when using complex codes that are time and ressources consuming. This is even more true in LES context as we are trying to have a representative simulation. The only solution to overcome this issue is to construct a model that would estimate a given quantity of interest (QoI) in a given range. This model requires a realistic number of evaluation of the detail code. Thus, the main purpose of JPOD is to construct a surrogate model from a complex code [Thie2009]_. This code can either be a simple *1D* function or a complex code like AVBP. The general procedure to construct it consists in:

* Generate a sample space:
    Also called design of experiment (DOE), it consists in generating a set of data, with good space filling properties as discussed by Damblin et al. in [Damblin2013]_, from which to run the code. A solution is called a *snapshot*.

* Learn the link between the input the output data:
    From the previously generated set of data, we can compute a model also called a response surface. A model is build using gaussian process [Rasmussen2006]_.

* Predict a solution from a new set of input data:
    The model can finaly be used to interpolate a new snapshot from a new set of input data.

Once this model has been constructed, using Monte Carlo sampling we can compute Sobol' indices etc.

Both **POD** and **Kriging** are techniques that can interpolate data using snapshots. The main difference being that POD compresses the data it uses using only the relevant modes wherease Kriging method doesn't reduce the size of the used snapshots. On the other hand, POD connot reconstruct data from a domain missing ones [Gunes2006]_. Thus, the strategy used by JPOD consists in:

1. Use POD reconstruction in order to compress data,
2. Use Kriging interpolation on POD's coefficients,
3. Interpolate missing data.


.. seealso:: More details about DOE, GP


Content of the JPOD package
---------------------------

The JPOD package includes 2 repository:

* ``kernel`` contains the JPOD package and its implementation,
* ``test_cases`` contains some example.


General functionment of the tool
................................

The package is composed of several python modules which are self contained within the directory ``kernel/jpod``.
Following is a quick reference:

* :py:mod:`ui`: user interface,
* :py:mod:`driver`: contains the main functions,
* :py:mod:`uq`: uncertainty quantification,
* :py:mod:`algebra`: constructs the surrogate model,
* :py:mod:`space`: defines the (re)sampling space,
* :py:mod:`pod`: constructs the POD,
* :py:mod:`tasks`: 
* :py:mod:`misc`: define 

The module :py:mod:`ui` is the main script. It takes one argument which is the task to perform::

    python ~/JPOD/kernel/jpod/ui.py task.py

The latter loads the context to compute each snapshot from. The tool then creates an ``output`` folder which will contain the results of the computation of all the *snapshots*, the *pod* and the *predictions*.

.. image:: fig/UML.pdf

Content of ``test_cases``
.........................


.. [Thie2009] T. Braconnier and M. Ferrier: Jack Proper Orthogonal Decomposition (JPOD) for Steady Aerodynamic Model. Tech. rep. 2009
.. [Rasmussen2006] CE. Rasmussen and C. Williams: Gaussian processes for machine learning. MIT Press. 2006. ISBN: 026218253X
.. [Damblin2013] G. Damblin, M. Couplet, B. Iooss: Numerical studies of space filling designs : optimization of Latin Hypercube Samples and subprojection properties. Journal of Simulation. 2013.
.. [Gunes2006] H. Gunes, S. Sirisup and GE. Karniadakis: “Gappydata:ToKrigornottoKrig?”. Journal of Com putational Physics. 2006. DOI: 10. 1016/j.jcp.2005.06.023