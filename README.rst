|CI|_ |Codecov|_ |Python|_ |License|_ |Zulip|_

.. |CI| image:: https://gitlab.com/cerfacs/batman/badges/develop/pipeline.svg
   .. _CI: https://gitlab.com/cerfacs/batman/pipelines

.. |Codecov| image:: https://gitlab.com/cerfacs/batman/badges/develop/coverage.svg
   .. _Codecov: https://gitlab.com/cerfacs/batman/pipelines

.. |Python| image:: https://img.shields.io/badge/python-2.7,_3.6-blue.svg

.. |License| image:: https://img.shields.io/badge/license-CECILL--B_License-blue.svg
   .. _License: http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

.. |Zulip| image:: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
   .. _Zulip: https://batman-cerfacs.zulipchat.com

===========
Quick Start
===========

Introduction
------------

**BATMAN** stands for Bayesian Analysis Tool for Modelling and uncertAinty
quaNtification. It is a Python module distributed under the open-source
CECILL-B license (MIT/BSD compatible).

BATMAN stands for Bayesian Analysis Tool for Modelling and uncertAinty quaNtification. It aims at:

- Building Surrogate Models (Gaussian process, Polynomial Chaos, RBF),
- Optimization (Expected Improvement),
- Reucting the data with the help of POD (Proper Orthogonal Decomposition),
- Optimizing the number of huge computations by using advanced Design of Experiment techniques,
- Automatically managing code computations in parallel,
- Realizing Sensitivity Analysises (SA) and Uncertainty Quantifications (UQ),
- Visualization in high dimensions (HDR, Kiviat).

A full documentation is available at: https://cerfacs.gitlab.io/batman

How to get BATMAN?
------------------

The sources are located on the *GitLab* server: 

https://gitlab.com/cerfacs/batman


How to install BATMAN?
----------------------

Dependencies
............

- `Python <https://python.org>`_ >= 2.7 or >= 3.3
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
- `ffmpeg <https://www.ffmpeg.org>`_ for movie visualizations

Testing dependencies are: 

- `pytest <https://docs.pytest.org/en/latest/>`_ >= 2.8
- `mock <https://pypi.python.org/pypi/mock>`_ >= 2.0

Extra testing flavours: 

- `pytest-cov <https://github.com/pytest-dev/pytest-cov>`_ >= 2.5.1
- `pylint <https://www.pylint.org>`_ >= 1.6.0

.. note:: OpenTURNS and ffmpeg are available on *conda* through
    the *conda-forge* channel.

User Installation
.................

Using the latest python version is prefered! Then to install::

    git clone git@nitrox.cerfacs.fr:open-source/batman.git 
    cd batman
    python setup.py build_fortran
    python setup.py install
    python setup.py test
    python setup.py build_sphinx

The latter is optionnal as it build the documentation. The testing part is also
optionnal but is recommanded. (<30mins).

.. note:: If you don't have install priviledge, add ``--user`` option after install.
    But the simplest way might be to use a conda environment.

Finally, if you want to install the optionnal package ``Antares``::

    pip install --editable .[antares] --process-dependency-links

If batman has been correctly installed, you should be able to call it simply::

    batman -h

.. warning:: Depending on your configuration, you might have to export your local path: 
    ``export PATH=$PATH:~/.local/bin``. Care to be taken with both your ``PATH``
    and ``PYTHONPATH`` environment variables. Make sure you do not call different
    installation folders. It is recommanded that you leave your ``PYTHONPATH`` empty.

Otherwize (if you want Python 3 for instance) you can create your ``conda`` environment::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    conda create -n bat_env -c conda-forge openturns matplotlib numpy scipy scikit-learn pathos jsonschema sphinx sphinx_rtd_theme pytest pytest-runner mock ffmpeg

Then you can install all packages without ``root`` access. You can access
the newly created environment with ``source activate bat_env``.

.. note:: All changes can be found in :ref:`changes`. The main folder contains three
 subfolders: ``doc`` ``batman`` and ``test_cases``. The latter contains examples that you can adapt to you needs. You can find more information about the cases within the respectives ``README.rst`` file. A detailled example can be found in :ref:`tutorial`.

Help and Support
----------------

If you encounter a bug (or have a feature request), please report it via
`GitLab <https://gitlab.com/cerfacs/batman/issues>`_

A HTML documentation is available https://cerfacs.gitlab.io/batman                       

Communication
-------------

- IRC channel: ``#batman`` at ``cerfacs.slack.com``

Citation
--------

If you use batman in a scientific publication, we would appreciate :ref:`citations <citing-batman>`.
