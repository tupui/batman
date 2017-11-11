|CI|_ |Codecov|_ |Python|_ |License|_ |Zulip|_

.. |CI| image:: https://nitrox.cerfacs.fr/open-source/batman/badges/develop/build.svg
.. _CI: https://nitrox.cerfacs.fr/open-source/batman/pipelines

.. |Codecov| image:: https://nitrox.cerfacs.fr/open-source/batman/badges/develop/coverage.svg
.. _Codecov: https://nitrox.cerfacs.fr/open-source/batman/pipelines

.. |Python| image:: https://img.shields.io/badge/python-2.7,_3.6-blue.svg

.. |License| image:: https://img.shields.io/badge/license-CECILL--B_License-blue.svg
.. _License: http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

.. |Zulip| image:: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
.. _Zulip: https://batman-cerfacs.zulipchat.com

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
- Metamodel (Gaussian process, Polynomial Chaos, RBF),
- Optimization (Expected Improvement),
- Visualize both sample and CFD computations in *n*-dimensions (HDR, Kiviat),
- *POD* for database optimization or data reduction,
- Automatically manage (parallel) the numerical computations.

Getting started
---------------

A detailled example can be found in 
`tutorial <http://open-source.pg.cerfacs.fr/batman/tutorial.html>`_ and the
full documentation is available at: 

    http://open-source.pg.cerfacs.fr/batman

The main folder contains three subfolders: ``doc`` ``batman`` and ``test_cases``.
The latter contains examples that you can adapt to you needs. You can find more
information about the cases within the respectives ``README.rst`` file. 

Shoud you be interested by batman's implementation, consider
reading `introduction <http://open-source.pg.cerfacs.fr/batman/introduction.html>`_.

If you encounter a bug (or have a feature request), please report it via
`GitLab <https://gitlab.com/cerfacs/batman>`_. Or it might be you
falling but "Why do we fall sir? So we can learn to pick ourselves up".

Last but not least, if you consider contributing check-out
`contributing <http://open-source.pg.cerfacs.fr/batman/contributing.html>`_.

Happy batman.

How to get it?
--------------

The sources are located on *GitLab*: 

    https://gitlab.com/cerfacs/batman

How to Install?
---------------

Dependencies
............

The required dependencies are: 

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

User installation
.................

Using the latest python version is prefered! Then to install::

    git clone git@nitrox.cerfacs.fr:open-source/batman.git 
    cd batman
    python setup.py build_fortran
    python setup.py install
    python setup.py test
    python setup.py build_sphinx

The latter is optionnal as it build the documentation. The testing part is also
optionnal but is recommanded. (<30mins depending on your configuration).

.. note:: If you don't have install priviledge, add ``--user`` option after install.
    But the simplest way might be to use a conda environment.

Finally, if you want to install the optionnal package ``Antares`` (not provided)::

    pip install --editable .[antares] --process-dependency-links

If batman has been correctly installed, you should be able to call it simply::

    batman -h

.. warning:: Depending on your configuration, you might have to export your local path: 
    ``export PATH=$PATH:~/.local/bin``. Care to be taken with both your ``PATH``
    and ``PYTHONPATH`` environment variables. Make sure you do not call different
    installation folders. It is recommanded that you leave your ``PYTHONPATH`` empty.

Otherwize you can create your ``conda`` environment::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    conda create -n bat_env -c conda-forge openturns matplotlib numpy scipy scikit-learn pathos jsonschema sphinx sphinx_rtd_theme pytest pytest-runner mock ffmpeg

Then you can install all packages without ``root`` access. You can access
the newly created environment with ``source activate bat_env``.

Help and Support
----------------

About us
........

See authors and project history at: `about us <http://open-source.pg.cerfacs.fr/batman/about.html>`_.

Community
.........

If you use batman, come and say hi at https://batman-cerfacs.zulipchat.com.
Or send us an email. We would really appreciate that as we keep record of the users!

Citation
........

If you use batman in a scientific publication, we would appreciate `citations <http://open-source.pg.cerfacs.fr/batman/about.html#citing-batman>`_.
