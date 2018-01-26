.. _changes:

.. currentmodule:: batman

===============
Release history
===============

Version 1.7.3 - Lucius
======================

.. note:: This version includes all comments from JOSS' reviewers. See the 
   `review <https://github.com/openjournals/joss-reviews/issues/493#issuecomment-360492888>`_.

New features
------------

    - Refactor :mod:`input_output`. Remove fortran and greatly simplify IO
      handling. *by Cyril Fournier*,
    - Add ``extremum`` option in settings for resampling, *by Pamphile Roy*,
    - Add :class:`surrogate.SklearnRegressor` as an interface to all Scikit-Learn
      regressors. Available through ``method`` in surrogate's settings.
      *by Pamphile Roy*.

Enhancements
------------

    - Do not compute quality for optimization and discrepancy, *by Pamphile Roy*,
    - Reduce bounds amplitude and add warning for convergence, *by Pamphile Roy*.

Bug fixes
---------

    - Remove documentation from gitlab pages, *by Pamphile Roy*.


Version 1.7.2 - Lucius
======================

New features
------------

    - Refactor :class:`tasks.snapshot`. Settings have been simplified and
      code maintenance has been eased. *by Cyril Fournier*,
    - Add new visualization :class:`visualization.Tree` for 2D, *by Pamphile Roy*,
    - Add ``global_optimizer`` option in settings for Kriging, *by Pamphile Roy*,
    - Move documentation to *read the docs*, *by Pamphile Roy*,
    - Add documentation for MASCARET and PCE, *by Matthias De Lozzo*.

Enhancements
------------

    - Export :class:`visualization.Kiviat` as a mesh, *by Pamphile Roy*,
    - Some visualization for MASCARET, *by Sophie Ricci*.

Bug fixes
---------

    - Point mixing when snapshots already exists, *by Pamphile Roy*,
    - Outlier are computed only once in f-HOPs, *by Pamphile Roy*,
    - PDF scaling, *by Pamphile Roy*,
    - Legend list for forward compatibility with MPL, *by Pamphile Roy*,
    - Range color bar, visualization ticks default, *by Pamphile Roy*,
    - Driver exceptions, *by Pamphile Roy*,
    - Encoding errors in schema with python 3.5, *by Pamphile Roy*,
    - Settings checking was not effective, *by Pamphile Roy*.


Version 1.7.1 - Lucius
======================

New features
------------

    - Add a ``fill`` option in :class:`visualization.Kiviat`, *by Pamphile Roy*,
    - Add ``bounds`` option in visualization settings, *by Pamphile Roy*,
    - :class:`visualization.Kiviat` automatically used by driver if dim > 4, *by Robin Campet*,
    - Allow duplicate points in :class:`space.Space`, *by Pamphile Roy*.

Enhancements
------------

    - Visualization settings taken into account for PDF, legend outside, *by Pamphile Roy*,
    - Refactor :class:`space.Space` error handling, *by Pamphile Roy*,
    - Documentation reorganization, *by Jean-Christophe Jouhaud*.

Bug fixes
---------

    - :class:`visualization.Kiviat` filling and ordering, *by Pamphile Roy*,
    - Maths in documentation as PNG, *by Pamphile Roy*,
    - Projection strategy in :class:`surrogate.PC`, *by Pamphile Roy*,
    - Circular imports from :func:`functions.utils.multi_eval`, *by Sophie Ricci*,
    - Variance in LOO Q2, *by Romain Dupuis*,
    - :class:`surrogate.PC` restart with *LS* strategy, *by Andrea Trucchia*.


Version 1.7 - Lucius
====================

New features
------------

    - Add :func:`space.Space.discrepancy` function in ``Space``, *by Pamphile Roy*,
    - Refactor :class:`space.Space`, :class:`uq.UQ`, *by Pamphile Roy*,
      :class:`space.Refiner` initiate without dictionnary, *by Pamphile Roy*,
    - Refactor :class:`surrogate.PC` and add options in
      settings: ``degree``, ``strategy`` (*Quad* or *LS*), *by Pamphile Roy*,
    - Add :func:`space.Refiner.discrepancy`, and
      :func:`space.Refiner.sigma_discrepancy`, *by Pamphile Roy*,
    - Add quality for every surrogate model, *by Pamphile Roy*,
    - Be able to bypass POD and surrogate in settings, *by Pamphile Roy*,
    - Surrogate facultative for UQ, *by Pamphile Roy*,
    - Add :mod:`visualization` with: Kiviat, DoE, HDR *by Pamphile Roy*,
    - and response_surface with block ``visualization`` in settings, *by Robin Campet*,
    - Add ``distributions`` in settings to set a distribution per parameter, *by Pamphile Roy*,
    - Add ``discrete`` in settings to tell the indice of the discrete paramter, *by Pamphile Roy*,
    - Add :class:`functions.Data` for datasets with some new ones,
    - Add optimized LHS, *by Vincent Baudoui*,
    - Add noise and kernel for Kriging in settings, *by Andrea Trucchia*,
    - Header is now a JSON file, *by Cyril Fournier*,
    - Concurrent CI, *by Cyril Fournier*,
    - pylint/pycodestyle for CI and Python2 on develop and master branches, *by Pamphile Roy*,
    - Add *about* section in doc, *by Pamphile Roy*.

Enhancements
------------

    - Remove loops in predictors, ``zip``, *by Pamphile Roy*,
    - Backend overwright for matplotlib removed, *by Pamphile Roy*,
    - Remove ``otwrapy``, *by Pamphile Roy*,
    - JSON schema constrained for surrogate and sampling, *by Pamphile Roy*,
    - Refactor :class:`pod.Pod`, *by Pamphile Roy*,
    - Sobol' indices with ensemble, *by Pamphile Roy*,
    - Remove support for OpenTURNS < 1.8, *by Pamphile Roy*,
    - Add some options for :class:`functions.MascaretApi`, *by Pamphile Roy*,
    - Coverage and tests raised to 90%, *by Pamphile Roy*.

Bug fixes
---------

    - Quality with multimodes with POD, *by Pamphile Roy*,
    - List in sampling settings, *by Pamphile Roy*,
    - Restart and restart from files, *by Pamphile Roy*,
    - Other file read with restart, *by Cyril Fournier*,
    - Variance and FAST, *by Pamphile Roy*,
    - Double prompt in python 2.7, *by Vincent Baudoui*,
    - DoE as list, *by Vincent Baudoui*,
    - Inputs mocking in tests, *by Pamphile Roy*,
    - DoE diagonal scaling, *by Pamphile Roy*,
    - :class:`functions.MascaretApi` ``multi_eval``, *by Pamphile Roy*,
    - Block indices, *by Pamphile Roy*,
    - Installation without folder being a git repository, *by Cyril Fournier*,
    - Fortran compilation, *by Cyril Fournier*,
    - Normalize output in :class:`surrogate.Kriging`, *by Pamphile Roy*.


Version 1.6 - Selina
====================

New features
------------

    - Add :class:`functions.MascaretApi`, *by Pamphile Roy*,
    - Add *Evofusion* with :class:`surrogate.Evofusion`, *by Pamphile Roy*,
    - Add *Expected Improvement* with :func:`space.Refiner.optimization`, *by Pamphile Roy*,
    - Be able to have a discrete parameter, *by Pamphile Roy*.

Enhancements
------------

    - Allow ``*args`` and ``**kwargs`` in ``@multi_eval``, *by Pamphile Roy*,
    - Add some analytical functions for optimization and multifidelity tests, *by Pamphile Roy*,
    - Do not use anymore ``.size`` for space sizing, *by Pamphile Roy*,
    - Add test for DoE, *by Pamphile Roy*,
    - Add PDFs of references to doc, *by Pamphile Roy*,
    - Refinements methods work with discrete values using an optimizer decorator, *by Pamphile Roy*,
    - Changed some loops in favor of list comprehensions, *by Pamphile Roy*,
    - Clean UI by removing prediction option, *by Pamphile Roy*,
    - Remove MPI dependencie, *by Pamphile Roy*.

Bug fixes
---------

    - Sensitivity indices with n-dimensional output changing ``Martinez``, *by Pamphile Roy*,
    - A copy of the space is done for scaled points for surrogate fitting, *by Pamphile Roy*,
    - Uniform sampling was not set properly, *by Pamphile Roy*,
    - Backend for ``matplotlib`` is now properly switched, *by Pamphile Roy*,
    - POD quality was not computed in case of varying number of modes, *by Pamphile Roy*.


Version 1.5 - Oswald
====================

New features
------------

    - Python3 support, *by Pamphile Roy*,
    - Add :class:`surrogate.surrogate_model.SurrogateModel`, *by Pamphile Roy*,
    - Add progress bar during quality computation, *by Pamphile Roy*,
    - Use pathos for multiprocessing during LOO and Kriging.
      New :class:`misc.nested_pool` allow nested pool. *by Pamphile Roy*,
    - Unittests and functionnal tests using Pytest, *by Pamphile Roy*,
    - Antares wrapper used for IO, *by Pamphile Roy*,
    - OT1.8 support and use of new SA classes, *by Pamphile Roy*,
    - Add plot of aggregated indices, *by Pamphile Roy*,
    - Add *snipets*, *by Pamphile Roy*,
    - Add correlation and covariance matrices, *by Pamphile Roy*,
    - Add DoE visualization in *n-dimension*, *by Pamphile Roy*,
    - Hypercube for refinement created using discrete and global optimization, *by Pamphile Roy*,
    - Merge some ``PyUQ`` functions and add :class:`surrogate.PC`, *by Pamphile Roy*.
    
Enhancements
------------

    - Refactor :mod:`space`, :mod:`surrogate`, :mod:`snapshots`, :mod:`pod`, *by Pamphile Roy*,
    - Rewrite ``settings.json``, *by Pamphile Roy*,
    - POD is now optional, *by Pamphile Roy*,
    - Use a wrapper for OT evaluations with ``otwrapy``, *by Pamphile Roy*,
    - Comment capability to ``settings.json``, *by Pamphile Roy*,
    - Doc cleanning, *by Pamphile Roy*,
    - Use :mod:`functions` to test model error, *by Pamphile Roy*,
    - Remove some MPI functions, *by Pamphile Roy*,
    - Simplify hybrid navigator using generator, *by Pamphile Roy*.

Bug fixes
---------

    - Use of timeout option, *by Pamphile Roy*,
    - Remove ``snapshots.tar``, *by Pamphile Roy*,
    - FAST indices for aggregated indices, *by Pamphile Roy*,
    - Update keyword for POD, *by Pamphile Roy*,
    - Verbosity with quality, *by Pamphile Roy*,
    - Setup dependencies, *by Pamphile Roy*,
    - Some RBF cleanning, *by Pamphile Roy*,
    - Term *MSE* changed to *sigma*, *by Pamphile Roy*,
    - Snapshot ``repr``, *by Pamphile Roy*,
    - Add *.so* when packaging, *by Pamphile Roy*.


Version 1.4
===========

New features
------------

    - Enhance :class:`surrogate.kriging`: adimentionize input parameters,
      use anisotropic kernel and use genetic algorithm for parameters optimization, *by Pamphile Roy*,
    - Settings are now written in JSON and checked using a schema, *by Pamphile Roy*,
    - Ask for confirmation of output if exists: if no, ask for restarting from files, *by Pamphile Roy*,
    - Add post-treatment example scripts in ``test_cases/Post-treatment``, *by Pamphile Roy*.

Enhancements
------------

    - Save points of the DOE as human readable file, *by Pamphile Roy*,
    - Add branch and commit information to log, *by Pamphile Roy*,
    - Add doc for tutorial, space, surrogate and pod, *by Pamphile Roy*,
    - Change Scikit-Learn to stable 0.18, *by Pamphile Roy*,
    - Restart option ``-r`` now working properly, *by Pamphile Roy*,
    - Create a :mod:`misc` which contains logging and json schema, *by Pamphile Roy*.

Bug fixes
---------

    - Refiner navigator loops correctly, *by Pamphile Roy*,
    - LOOCV working for multimodes, *by Pamphile Roy*,
    - Revert Q2 variance to use ``eval_ref``, *by Pamphile Roy*,
    - Avoid extra POD quality when using LOOCV strategies, *by Pamphile Roy*,
    - Popping space was not working properly, *by Pamphile Roy*.


Version 1.3
===========

New features
------------

    - Add resampling strategies with :class:`space.refiner`. Possibilities are:
      ``None, MSE, loo_mse, loo_sobol, hybrid``, *by Pamphile Roy*,
    - Computation of the error of the pod *Q2* with option ``-q2``. Uses *Kriging*, *by Pamphile Roy*,
    - Aggregated and block *Sobol'* indices are computed using a set of keywords:
      ``aggregated`` and ``block``, *by Pamphile Roy*,
    - Add the possibility to chose the *PDF* for propagation. (settings), *by Pamphile Roy*,
    - *Sobol'* map are computed using the keyword ``aggregated``, *by Pamphile Roy*,
    - A *Sphinx* documentation is available in: ``/doc``, *by Pamphile Roy*.

Enhancements
------------

    - Change command line interface parsing with :class:`argparse`.
      Also  remove ``--plot`` option and add output default repository, *by Pamphile Roy*,
    - Installation is more Pythonic has it uses now a ``setup.py`` script, *by Pamphile Roy*,
    - The project can be imported: ``import jpod``, *by Pamphile Roy*,
    - Settings are defined ones as an attribute of :class:`driver`, *by Pamphile Roy*,
    - Logger is now simpler and configuration can be changed prior installation in: ``/misc/logging.json``, *by Pamphile Roy*,
    - When defining a sample size for *UQ*, the value is used for indices and propagation, *by Pamphile Roy*,
    - The keyword ``pod['quality']`` correspond now to the targeted *Q2*, *by Pamphile Roy*,
    - Add *Python3* compatibility, *by Pamphile Roy*.

Bug fixes
---------

    - *Kriging* was not working with several modes, *by Pamphile Roy*,
    - Output folder for ``uq`` was not working, *by Pamphile Roy*,
    - ``NaN`` for uncertainty propagation, *by Pamphile Roy*,
    - Remove auto keyword from ``pod['type']``, *by Pamphile Roy*.


Version 1.2
===========

New features
------------

    - Add uncertainty quantification capability with :class:`uq` and the option ``-u``.
      ``sobol`` or ``FAST`` indices are computed on a defined sample size.
      Configuration is done within settings dictionnary file. Test functions are available.
      An output folder ``uq`` is created and contains indices and propagation data, *by Pamphile Roy*,
    - New test case ``Function_3D`` used to demonstrate *UQ* capabilities of the tool, *by Pamphile Roy*,
    - Sampling is now done using the package *OpenTURNS*, *by Pamphile Roy*,
    - New test case ``Channel_Flow`` used to demonstrate *1D vector* output capabilities, *by Pamphile Roy*,

Enhancements
------------

    - *Kriging* is now done using the module :mod:`sklearn.gaussian_process` from the package *Scikit-Learn*, *by Pamphile Roy*.
