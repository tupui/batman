.. image:: https://nitrox.cerfacs.fr/open-source/batman/badges/develop/build.svg
   :target: https://nitrox.cerfacs.fr/open-source/batman/pipelines
   :alt: build status

.. image:: https://nitrox.cerfacs.fr/open-source/batman/badges/develop/coverage.svg
   :target: https://nitrox.cerfacs.fr/open-source/batman/pipelines
   :alt: coverage status

.. image:: https://img.shields.io/badge/python-2.7,_3.6-blue.svg

.. image:: https://img.shields.io/badge/release-v1.6_Selina-blue.svg


BATMAN
======

**BATMAN** stands for Bayesian Analysis Tool for Modelling and uncertAinty quaNtification.
It aims at:

- Build a metamodel for design, optimization and database exchange (loads, MDO, identification),
- Can use a *POD* for database optimization or data reduction,
- Construct model for the full field (conservative variables, local quantities…),
- Optimize the number of CFD computations,
- Automatically manage (parallel) the CFD computations,
- Have the optimal representation of the problem for a minimal cost in term of CFD evaluations.

Aside from that, an uncertainty quantification (UQ) module allows to make
sensitivity analysis (SA) and uncertainty propagation.

Getting started
---------------

All changes can be found in :ref:`changes`. The main folder contains three
subfolders: ``doc`` ``batman`` and ``test_cases``. The latter contains examples
that you can adapt to you needs. You can find more information about the cases
within the respectives ``README.rst`` file. A detailled example can be found in
:ref:`tutorial`.

Aside from the documentation folder, the HTML documentation is available
`here <http://open-source.pg.cerfacs.fr/batman/>`_.

Shoud you be interested by batman's implementation, consider
reading :ref:`introduction`.

If you encounter a bug (or have a feature request), please report it via
`GitLab <https://nitrox.cerfacs.fr/open-source/batman>`_. Or it might be you
falling but "Why do we fall sir? So we can learn to pick ourselves up".

Last but not least, if you consider contributing check-out :ref:`contributing`.

Happy batman.

How to get it?
--------------

The sources are located on the *GitLab* server: 

    https://nitrox.cerfacs.fr/open-source/batman

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

Help and Support
----------------

About us
........

See authors and project history at: :ref:`about`.

Communication
.............

- IRC channel: ``#batman`` at ``cerfacs.slack.com``

Citation
........

If you use batman in a scientific publication, we would appreciate :ref:`citations <citing-batman>`.
