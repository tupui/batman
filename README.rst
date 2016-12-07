README
======

:Authors: 
    Pamphile Roy,
    Romain Dupuis,
    Jean-Christophe Jouhaud,
    Marc Montagnac,
    Florent Duchaine,
    Jean-François Boussuge,
    Melanie Rochoux,
    Sophie Ricci.

:Version: 1.4   10/10/2016


What is JPOD? 
-------------

**JPOD** stands for Jack Proper Orthogonal Decomposition. It aims at:

- Use *POD* to build a metamodel for design, optimization and database exchange (loads, MDO, identification),
- Construct model for the full field (conservative variables, local quantities…),
- Optimize the number of CFD computations,
- Automatically manage (parallel) the CFD computations and the POD reduction,
- Have the optimal representation of the problem for a minimal cost in term of CFD evaluations.

Aside from that, an uncertainty quantification (UQ) module allows to make sensitivity analysis (SA) and uncertainty propagation.

Dependencies
------------

The required dependencies are: 

- Python >= 2.7 or >= 3.3
- `scikit-learn <http://scikit-learn.org>`_ >= 0.18
- `OpenTURNS <http://www.openturns.org>`_ >= 1.7
- `scipy <http://scipy.org>`_ >= 0.15

Optionnal dependencies are: 

- `Antares <http://www.cerfacs.fr/antares>`_
  
Appart from OpenTURNS and Antares, dependencies are satisfied by the installer.

How to get JPOD
---------------

You must belong to the ``uqs`` Unix group.

To download it::

    git clone ssh://dogon.cerfacs.fr/home/jpod_home/jpod/JPOD

Then to install::

    cd JPOD
    python setup.py build_fortran
    python setup.py install
    python setup.py build_sphinx

The latter is optionnal as it just build the documentation in case of a change. 

.. note:: If you don't have install priviledge, add ``--user`` option.

If JPOD has been correctly installed, you should be able to call it simply::

    jpod -h

.. note:: Depending on your configuration, you might have to export your local python path: 
 ``export PATH=$PATH:.../Python/2.7/bin``.

.. note:: OpenTURNS 1.7 is installed on *NEMO* for Python 2.7::

        module load python/2.7
        module load python/2.7-shared
        module load application/openturns/1.7

    The last version of OpenTURNS can be loaded using::

        module load python/miniconda2.7

    .. warning:: You cannot load both versions at the same time.

    Otherwize you can create your ``conda`` environment::

        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        conda create -n jpod_env -c conda-forge openturns

    Then you can install all packages without ``root`` access.

Getting started
---------------

All changes can be found in the :ref:`changes`. ``JPOD`` folder contrains three subfolders: ``doc`` ``jpod`` and ``test_cases``. The latter contains examples that you can adapt to you needs. You can find more information about the cases within the respectives ``README.rst`` file. A detailled example can be found in :ref:`tutorial`. Shoud you be interested by JPOD's implementation, consider reeding :ref:`introduction`.

Development Model
-----------------

Python
......

All developers must follow guidelines from the Python Software Foundation. As a quick reference:

* For text: `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
* For documentation: `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_
* Use reStructuredText formatting: `PEP 287 <https://www.python.org/dev/peps/pep-0287/>`_

And for a more Pythonic code: `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_

GIT
...

You can find the development model at http://nvie.com/posts/a-successful-git-branching-model/
Please **read** this page and **stick** to it.
The master and develop branches are dedicated to the manager only.
Release and hotfix branches are mandatory. They must be sent to the manager only in a **finished** state.

