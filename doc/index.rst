Welcome to BATMAN's documentation!
==================================

|CI|_ |Codecov|_ |Python|_ |License|_ |Zulip|_

.. |CI| image:: https://gitlab.com/cerfacs/batman/badges/develop/pipeline.svg
.. _CI: https://gitlab.com/cerfacs/batman/pipelines

.. |Codecov| image:: https://gitlab.com/cerfacs/batman/badges/develop/coverage.svg
.. _Codecov: https://gitlab.com/cerfacs/batman/pipelines

.. |Python| image:: https://img.shields.io/badge/python-2.7,_3.6-blue.svg
.. _Python: https://python.org

.. |License| image:: https://img.shields.io/badge/license-CECILL--B_License-blue.svg
.. _License: http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

.. |Zulip| image:: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
.. _Zulip: https://batman-cerfacs.zulipchat.com


BATMAN
======

Introduction
------------

**BATMAN** stands for Bayesian Analysis Tool for Modelling and uncertAinty
quaNtification. It is a Python module distributed under the open-source
CECILL-B license (MIT/BSD compatible).

batman seamlessly allows to do statistical analysis (sensitivity analysis,
Uncertainty Quantification, moments) using any computer solver.

Main features are: 

- Design of Experiment (LHS, low discrepancy sequences, MC),
- Resample the parameter space based on the physic and the sample,
- Surrogate Models (Gaussian process, Polynomial Chaos, RBF),
- Optimization (Expected Improvement),
- Realizing Sensitivity Analysises (SA) and Uncertainty Quantifications (UQ),
- Visualization in *n*-dimensions (HDR, Kiviat),
- *POD* for database optimization or data reduction,
- Automatically managing code computations in parallel.

Full documentation is available at: 

    https://cerfacs.gitlab.io/batman

Contents
--------

.. toctree::
   :maxdepth: 1

   Quick Start <quick_start>
   tutorial
   cli
   technical
   api
   contributing_link
   changes_link
   about

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About
-----

See :ref:`about`.
