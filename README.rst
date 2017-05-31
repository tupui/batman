README
======

:Authors: 
    Pamphile Roy,
    Romain Dupuis,
    Jean-Christophe Jouhaud,
    Marc Montagnac,
    Jean-François Boussuge,
    Florent Duchaine,
    Melanie Rochoux,
    Sophie Ricci.

:Version: 1.6 - Selina 01/06/2017


What is BATMAN? 
---------------

**BATMAN** stands for Bayesian Analysis Tool for Modelling And uNcertainty quantification.
It aims at:

- Build a metamodel for design, optimization and database exchange (loads, MDO, identification),
- Can use a *POD* for database optimization or data reduction,
- Construct model for the full field (conservative variables, local quantities…),
- Optimize the number of CFD computations,
- Automatically manage (parallel) the CFD computations,
- Have the optimal representation of the problem for a minimal cost in term of CFD evaluations.

Aside from that, an uncertainty quantification (UQ) module allows to make sensitivity analysis (SA) and uncertainty propagation.

Dependencies
------------

The required dependencies are: 

- Python >= 2.7 or >= 3.3
- `scikit-learn <http://scikit-learn.org>`_ >= 0.18
- `scipy <http://scipy.org>`_ >= 0.15
- `OpenTURNS <http://www.openturns.org>`_ >= 1.7
- `pathos <https://github.com/uqfoundation/pathos>`_ >= 0.2
- `otwrapy <http://openturns.github.io/otwrapy/>`_ >= 0.6
- `jsonschema <http://python-jsonschema.readthedocs.io/en/latest/>`_
- `sphinx <http://www.sphinx-doc.org>`_ >= 1.4

Optionnal dependencies are: 

- `Antares <http://www.cerfacs.fr/antares>`_
  
Appart from OpenTURNS and Antares, dependencies are satisfied by the installer.

How to get it?
--------------

The sources are located on the *GitLab* server: 

    https://nitrox.cerfacs.fr/open-source/batman

How to Install?
---------------

Then to install::

    cd BATMAN
    python setup.py build_fortran
    python setup.py install
    python setup.py test
    python setup.py build_sphinx

The latter is optionnal as it build the documentation.
The testing part is also optionnal but is recommanded. (<15mins).

.. note:: If you don't have install priviledge, add ``--user`` option after install.

Finally, to install the optionnal package ``Antares``::

    pip install -e .[antares] --process-dependency-links

If BATMAN has been correctly installed, you should be able to call it simply::

    batman -h

.. warning:: Depending on your configuration, you might have to export your local path: 
 ``export PATH=$PATH:~/.local/bin``.

.. note:: If using *NEMO* with Python 2.7::

        module load python/2.7
        module load python/2.7-shared
        module load application/openturns/1.7

    The last version of OpenTURNS can be loaded **after install** using instead::

        module load python/2.7
        module load python/2.7-shared
        module load python/miniconda2.7

    .. warning:: You cannot load both OpenTURNS versions at the same time.

    Otherwize (if you want Python 3 for instance) you can create your ``conda`` environment::

        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        conda create -n bat_env -c conda-forge openturns numpy scipy scikit-learn pathos jsonschema sphinx sphinx_rtd_theme otwrapy pytest pytest-runner mock

    Then you can install all packages without ``root`` access.

Getting started
---------------

All changes can be found in the :ref:`changes`. The main folder contains three
subfolders: ``doc`` ``batman`` and ``test_cases``. The latter contains examples
that you can adapt to you needs. You can find more information about the cases
within the respectives ``README.rst`` file. A detailled example can be found in
:ref:`tutorial`. Shoud you be interested by BATMAN's implementation, consider
reading :ref:`introduction`.

If you encounter a bug (or have a feature request), report it via `GitLab <https://nitrox.cerfacs.fr/open-source/batman>`_. Or it might be you falling but "Why do we fall sir? So we can learn to pick ourselves up".

Last but not least, if you consider contributing check-out :ref:`contributing`.

Happy BATMAN.
