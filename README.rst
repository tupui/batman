|CI|_ |Python|_ |License|_ |Joss|_

.. |CI| image:: https://circleci.com/gh/tupui/batman.svg?style=svg
.. _CI: https://circleci.com/gh/tupui/batman

.. |Python| image:: https://img.shields.io/badge/python-3.8-blue.svg
.. _Python: https://python.org

.. |License| image:: https://img.shields.io/badge/license-BSD_License-blue.svg
.. _License: https://github.com/tupui/batman/blob/master/LICENSE

.. |Joss| image:: https://joss.theoj.org/papers/a1c4bddc33a1d8ab55fce1a3596196d8/status.svg
.. _Joss: https://joss.theoj.org/papers/a1c4bddc33a1d8ab55fce1a3596196d8

Batman
======

**Batman** stands for Bayesian Analysis Tool for Modelling and uncertAinty
quaNtification. It is an open-source Python module.

*batman* seamlessly allows to do statistical analysis (sensitivity analysis,
Uncertainty Quantification, moments) based on non-intrusive ensemble experiment
using any computer solver. It relies on open source python packages dedicated
to statistics (`OpenTURNS <http://www.openturns.org>`_ and
`scikit-learn <http://scikit-learn.org>`_).

Main features are: 

- Design of Experiment (LHS, low discrepancy sequences, MC),
- Resample the parameter space based on the physic and the sample,
- Surrogate Models (Gaussian process, Polynomial Chaos, RBF, *scikit-learn*'s regressors),
- Optimization (Expected Improvement),
- Sensitivity/Uncertainty Analysis (SA, UA) and Uncertainty Quantification (UQ),
- Visualization in *n*-dimensions (HDR, Kiviat, PDF),
- *POD* for database optimization or data reduction,
- Automatically manage code computations in parallel.

Full documentation is available at: 

    http://batman.readthedocs.io

.. inclusion-marker-do-not-remove

Getting started
===============

A detailled example can be found in 
`tutorial <http://batman.readthedocs.io/en/latest/tutorial.html>`_. The folder ``test_cases``
contains examples that you can adapt to you needs. You can find more information
about the cases within the respectives ``README.rst`` file. 

Shoud you be interested by batman's implementation, consider
reading the `technical documentation <http://batman.readthedocs.io/en/latest/technical.html>`_.

If you encounter a bug (or have a feature request), please report it via
`GitLab <https://github.com/tupui/batman/issues>`_. Or it might be you
falling but "Why do we fall sir? So we can learn to pick ourselves up".

Last but not least, if you consider contributing check-out
`contributing <http://batman.readthedocs.io/en/latest/contributing_link.html>`_.

Happy batman.

How to install BATMAN?
----------------------

The sources are located on *GitLab*: 

    https://github.com/tupui/batman

Dependencies
............

The required dependencies are: 

- `Python <https://python.org>`_ >= 3.6
- `OpenTURNS <http://www.openturns.org>`_ >= 1.10
- `scikit-learn <http://scikit-learn.org>`_ >= 0.18
- `numpy <http://www.numpy.org>`_ >= 1.13
- `scipy <http://scipy.org>`_ >= 0.15
- `pathos <https://github.com/uqfoundation/pathos>`_ >= 0.2
- `matplotlib <http://matplotlib.org>`_ >= 2.1
- `Paramiko <http://www.paramiko.org>`_ >= 2.4
- `jsonschema <http://python-jsonschema.readthedocs.io/en/latest/>`_

Appart from OpenTURNS, required dependencies are satisfied by the installer.
Optionnal dependencies are: 

- `sphinx <http://www.sphinx-doc.org>`_ >= 1.4 for documentation
- `ffmpeg <https://www.ffmpeg.org>`_ for movie visualizations (*n_features* > 2)

Testing dependencies are: 

- `pytest <https://docs.pytest.org/en/latest/>`_ >= 2.8
- `mock <https://pypi.python.org/pypi/mock>`_ >= 2.0

Extra testing flavours: 

- `coverage <http://coverage.readthedocs.io>`_ >= 4.4
- `pylint <https://www.pylint.org>`_ >= 1.6.0

.. note:: OpenTURNS and ffmpeg are available on *conda* through
    the *conda-forge* channel.


From sources
............

Using the latest python version is prefered! Then to install::

    git clone git@github.com/tupui/batman.git
    cd batman
    python setup.py install
    python setup.py test
    python setup.py build_sphinx

The latter is optionnal as it build the documentation. The testing part is also
optionnal but is recommanded. (<30mins depending on your configuration).

.. note:: If you don't have install priviledge, add ``--user`` option after install.
    But the simplest way might be to use pip or a conda environment.

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

See authors and project history at: `about us <http://batman.readthedocs.io/en/latest/about.html>`_.

Community
.........

If you use batman, come and say hi at https://batman-cerfacs.zulipchat.com.
Or send us an email. We would really appreciate that as we keep record of the users!

Citation
........

If you use batman in a scientific publication, we would appreciate `citations <http://batman.readthedocs.io/en/latest/about.html#citing-batman>`_.
