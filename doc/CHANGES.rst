.. _changes:

===============
Release history
===============

Version 1.3
===========

New features
------------

    - Add resampling strategies with :class:`space.refiner`. Possibilities are:
      ``None, MSE, loo_mse, loo_sobol, hybrid``
    - Computation of the error of the pod *Q2* with option ``-q2``. Uses *Kriging*
    - Aggregated and block *Sobol'* indices are computed using a set of keywords:
      ``aggregated`` and ``block``
    - Add the possibility to chose the *PDF* for propagation. (settings)
    - *Sobol'* map are computed using the keyword ``aggregated``
    - A *Sphinx* documentation is available in: ``/doc``
    

Enhancements
------------

    - Change command line interface parsing with :class:`argparse`.
      Also  remove ``--plot`` option and add output default repository
    - Installation is more Pythonic has it uses now a ``setup.py`` script
    - The project can be imported: ``import jpod``
    - Settings are defined ones as an attribute of :class:`Driver`
    - Logger is now simpler and configuration can be changed prior installation in: ``/misc/logging.json``
    - When defining a sample size for *UQ*, the value is used for indices and propagation
    - The keyword ``pod['quality']`` correspond now to the targeted *Q2*
    - Add *Python3* compatibility

Bug fixes
---------

    - *Kriging* was not working with several modes
    - Output folder for ``uq`` was not working
    - ``NaN`` for uncertainty propagation
    - Remove auto keyword from ``pod['type']``


Version 1.2
===========

New features
------------

    - Add uncertainty quantification capability with :class:`uq` and the option ``-u``.
      ``sobol`` or ``FAST`` indices are computed on a defined sample size.
      Configuration is done within settings dictionnary file. Test functions are available.
      An output folder ``uq`` is created and contains indices and propagation data
    - New test case ``Function_3D`` used to demonstrate *UQ* capabilities of the tool
    - Sampling is now done using the package *OpenTURNS*
    - New test case ``Channel_Flow`` used to demonstrate *1D vector* output capabilities


Enhancements
------------

    - *Kriging* is now done using the module :mod:`sklearn.gaussian_process` from the package *Scikit-Learn*
