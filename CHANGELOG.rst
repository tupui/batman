.. _changes:

===============
Release history
===============

Version 1.5
===========

New features
------------

    - Python3 support,
    - Add :class:`batman.surrogate.surrogate_model`,
    - Add progress bar during quality computation,
    - Use pathos for multiprocessing during LOO and Kriging.
      New :class:`batman.misc.nested_popl` allow nested pool.
    - Unittests and functionnal tests using Pytest,
    - Antares wrapper used for IO,
    - OT1.8 support and use of new SA classes,
    - Add plot of aggregated indices,
    - Add *snipets*,
    - Add correlation and covariance matrices,
    - Add DoE visualization in *n-dimension*,
    - Hypercube for refinement created using discrete and global optimization,
    - Merge some ``PyUQ`` functions and add :class:`batman.surrogate.polynomial_chaos`.
    

Enhancements
------------

    - Refactor :mod:`batman.space`, :mod:`batman.predictor`, :mod:`batman.snapshots`, :mod:`batman.pod`,
    - Rewrite ``settings.json``,
    - POD is now optional,
    - Use a wrapper for OT evaluations with ``otwrapy``,
    - Comment capability to ``settings.json``,
    - Doc cleanning,
    - Use :mod:`batman.functions` to test model error,
    - Remove some MPI functions,
    - Simplify hybrid navigator using generator.

Bug fixes
---------

    - Use of timeout option,
    - Remove ``snapshots.tar``,
    - FAST indices for aggregated indices,
    - Update keyword for POD,
    - Verbosity with quality,
    - Setup dependencies,
    - Some RBF cleanning,
    - Term *MSE* changed to *sigma*,
    - Snapshot ``repr``,
    - Add *.so* when packaging.


Version 1.4
===========

New features
------------

    - Enhance :class:`batman.surrogate.kriging`: adimentionize input parameters,
      use anisotropic kernel and use genetic algorithm for parameters optimization
    - Settings are now written in JSON and checked using a schema
    - Ask for confirmation of output if exists: if no, ask for restarting from files
    - Add post-treatment example scripts in ``test_cases/Post-treatment``
    

Enhancements
------------

    - Save points of the DOE as human readable file
    - Add branch and commit information to log
    - Add doc for tutorial, space, surrogate and pod
    - Change Scikit-Learn to stable 0.18
    - Restart option ``-r`` now working properly
    - Create a :mod:`misc` which contains logging and json schema

Bug fixes
---------

    - Refiner navigator loops correctly
    - LOOCV working for multimodes
    - Revert Q2 variance to use ``eval_ref``
    - Avoid extra POD quality when using LOOCV strategies
    - Popping space was not working properly


Version 1.3
===========

New features
------------

    - Add resampling strategies with :class:`batman.space.refiner`. Possibilities are:
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
    - Settings are defined ones as an attribute of :class:`batman.driver`
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
