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

.. |Conda| image:: https://img.shields.io/badge/Install_with-conda-brightgreen.svg
.. _Conda: https://conda.anaconda.org/conda-forge/batman

.. |Joss| image:: https://joss.theoj.org/papers/a1c4bddc33a1d8ab55fce1a3596196d8/status.svg
.. _Joss: https://joss.theoj.org/papers/a1c4bddc33a1d8ab55fce1a3596196d8

BATMAN
======

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

.. inclusion-marker-do-not-remove

Getting started
===============

A detailled example can be found in 
`tutorial <https://cerfacs.gitlab.io/batman/tutorial.html>`_. The folder ``test_cases``
contains examples that you can adapt to you needs. You can find more information
about the cases within the respectives ``README.rst`` file. 

Shoud you be interested by batman's implementation, consider
reading `introduction <https://cerfacs.gitlab.io/batman/introduction.html>`_.

If you encounter a bug (or have a feature request), please report it via
`GitLab <https://gitlab.com/cerfacs/batman/issues>`_. Or it might be you
falling but "Why do we fall sir? So we can learn to pick ourselves up".

Last but not least, if you consider contributing check-out
`contributing <https://cerfacs.gitlab.io/batman/contributing_link.html>`_.

Happy batman.

How to install BATMAN?
----------------------

The sources are located on *GitLab*: 

    https://gitlab.com/cerfacs/batman

Dependencies
............

The required dependencies are: 

- `Python <https://python.org>`_ >= 2.7 or >= 3.4
- `scikit-learn <http://scikit-learn.org>`_ >= 0.18
- `numpy <http://www.numpy.org>`_ >= 1.13
- `scipy <http://scipy.org>`_ >= 0.15
- `OpenTURNS <http://www.openturns.org>`_ >= 1.9
- `pathos <https://github.com/uqfoundation/pathos>`_ >= 0.2
- `matplotlib <http://matplotlib.org>`_ >= 1.5
- `jsonschema <http://python-jsonschema.readthedocs.io/en/latest/>`_
- `sphinx <http://www.sphinx-doc.org>`_ >= 1.4

Appart from OpenTURNS, required dependencies are satisfied by the installer.
Optionnal dependencies are: 

- `Antares <http://www.cerfacs.fr/antares>`_ for extra IO options
- `ffmpeg <https://www.ffmpeg.org>`_ for movie visualizations (*n_features* > 2)

Testing dependencies are: 

- `pytest <https://docs.pytest.org/en/latest/>`_ >= 2.8
- `mock <https://pypi.python.org/pypi/mock>`_ >= 2.0

Extra testing flavours: 

- `coverage <http://coverage.readthedocs.io>`_ >= 4.4
- `pylint <https://www.pylint.org>`_ >= 1.6.0

.. note:: OpenTURNS and ffmpeg are available on *conda* through
    the *conda-forge* channel.

Latest release
..............

batman is distributed through ``conda``. To create a new environment which
contains batman simply::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    conda create -n bat_env -c conda-forge batman

You can access the newly created environment with ``source activate bat_env``.
All dependencies are automatically handled by ``conda``.

From sources
............

Using the latest python version is prefered! Then to install::

    git clone git@gitlab.com:cerfacs/batman.git
    cd batman
    python setup.py build_fortran
    python setup.py install
    python setup.py test
    python setup.py build_sphinx

The latter is optionnal as it build the documentation. The testing part is also
optionnal but is recommanded. (<30mins depending on your configuration).

.. note:: If you don't have install priviledge, add ``--user`` option after install.
    But the simplest way might be to use a conda environment.

If batman has been correctly installed, you should be able to call it simply::

    batman -h

.. warning:: Depending on your configuration, you might have to export your local path: 
    ``export PATH=$PATH:~/.local/bin``. Care to be taken with both your ``PATH``
    and ``PYTHONPATH`` environment variables. Make sure you do not call different
    installation folders. It is recommanded that you leave your ``PYTHONPATH`` empty.

Help and Support
----------------

About us
........

See authors and project history at: `about us <https://cerfacs.gitlab.io/batman/about.html>`_.

Community
.........

If you use batman, come and say hi at https://batman-cerfacs.zulipchat.com.
Or send us an email. We would really appreciate that as we keep record of the users!

Citation
........

If you use batman in a scientific publication, we would appreciate `citations <https://cerfacs.gitlab.io/batman/about.html#citing-batman>`_.