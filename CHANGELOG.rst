.. _changes:

===============
Release History
===============

Version 1.7 - Lucius (under development)
========================================

New features
------------

    - Add :func:`space.Space.discrepancy` function in ``Space``,
    - Refactor :class:`space.Space`, :class:`uq.UQ`,
      :class:`space.Refiner` initiate without dictionnary,
    - Refactor :class:`surrogate.PC` and add options in
      settings: ``degree``, ``strategy`` (*Quad* or *LS*),
    - Add :func:`space.Refiner.discrepancy`, and
      :func:`space.Refiner.sigma_discrepancy`
    - Add quality for every surrogate model,
    - Be able to bypass POD and surrogate in settings,
    - Surrogate facultative for UQ,
    - Add :mod:`visualization` with: Kiviat, DoE, HDR and response_surface,
      response_surface *by Robin Campet*,
    - Add optimized LHS, *by Vincent Baudoui*,
    - Concurrent CI, *by Cyril Fournier*,
    - pylint/pycodestyle for CI and Python2 on develop and master branches,
    - Add *about* section in doc.

Enhancements
------------

    - Remove loops in predictors, ``zip``,
    - Backend overwright for matplotlib removed,
    - Remove ``otwrapy``,
    - JSON schema constrained for surrogate and sampling,
    - Remove support for OpenTURNS < 1.8,
    - Add some options for :class:`functions.MascaretApi`,
    - Coverage and tests raised to 90%.

Bug fixes
---------

    - Quality with multimodes with POD,
    - List in sampling settings,
    - Restart and restart from files,
    - Variance and FAST,
    - Double prompt in python 2.7, *by Vincent Baudoui*,
    - Inputs mocking in tests,
    - DoE diagonal scaling,
    - :class:`functions.MascaretApi` ``multi_eval``,
    - Block indices,
    - Normalize output in :class:`surrogate.Kriging`.


Version 1.6 - Selina
====================

New features
------------

    - Add :class:`functions.MascaretApi`,
    - Add *Evofusion* with :class:`surrogate.Evofusion`,
    - Add *Expected Improvement* with :func:`space.Refiner.optimization`,
    - Be able to have a discrete parameter.

Enhancements
------------

    - Allow ``*args`` and ``**kwargs`` in ``@multi_eval``,
    - Add some analytical functions for optimization and multifidelity tests,
    - Do not use anymore ``.size`` for space sizing,
    - Add test for DoE,
    - Add PDFs of references to doc,
    - Refinements methods work with discrete values using an optimizer decorator,
    - Changed some loops in favor of list comprehensions,
    - Clean UI by removing prediction option,
    - Remove MPI dependencie.

Bug fixes
---------

    - Sensitivity indices with n-dimensional output changing ``Martinez``,
    - A copy of the space is done for scaled points for surrogate fitting,
    - Uniform sampling was not set properly,
    - Backend for ``matplotlib`` is now properly switched,
    - POD quality was not computed in case of varying number of modes.


Version 1.5 - Oswald
====================

New features
------------

    - Python3 support,
    - Add :class:`surrogate.surrogate_model.SurrogateModel`,
    - Add progress bar during quality computation,
    - Use pathos for multiprocessing during LOO and Kriging.
      New :class:`misc.nested_pool` allow nested pool.
    - Unittests and functionnal tests using Pytest,
    - Antares wrapper used for IO,
    - OT1.8 support and use of new SA classes,
    - Add plot of aggregated indices,
    - Add *snipets*,
    - Add correlation and covariance matrices,
    - Add DoE visualization in *n-dimension*,
    - Hypercube for refinement created using discrete and global optimization,
    - Merge some ``PyUQ`` functions and add :class:`surrogate.PC`.
    
Enhancements
------------

    - Refactor :mod:`space`, :mod:`surrogate`, :mod:`snapshots`, :mod:`pod`,
    - Rewrite ``settings.json``,
    - POD is now optional,
    - Use a wrapper for OT evaluations with ``otwrapy``,
    - Comment capability to ``settings.json``,
    - Doc cleanning,
    - Use :mod:`functions` to test model error,
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

    - Enhance :class:`surrogate.kriging`: adimentionize input parameters,
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
    - Settings are defined ones as an attribute of :class:`driver`
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
