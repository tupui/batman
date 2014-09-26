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

:Version: 1.3   20/07/2016


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
- `OpenTURNS <http://www.openturns.org>`_ == 1.7

.. note:: OpenTURNS is installed on *NEMO* only for Python 2.7  other dependencies are satisfied by JPOD's installer.

How to get JPOD
---------------

You must belong to the ``uqs`` Unix group.

To download it::

    git clone /home/jpod_home/jpod/JPOD
    git clone ssh://dogon.cerfacs.fr/home/jpod_home/jpod/JPOD

Then to install::

    cd kernel
    python setup.py build_fortran
    python setup.py install
    python setup.py build_sphinx

The latter is optionnal as it just build the documentation in case of a change.

Getting started
---------------

All changes can be found in the :ref:`changes`. ``JPOD`` folder contrains two subfolders: ``kernel`` and ``test_cases``. The latter contains examples that you can adapt to you needs. You can find more information about the cases within the respectives ``README.rst`` file. A detailled example can be found in :ref:`tutorial`. Shoud you be interested by JPOD's implementation, consider reeding :ref:`introduction`.

Development Model
-----------------

Python
......

All developers must follow guidelines from the Python Software Foundation.
As a quick reference:

* For text: `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
* For documentation: `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_
* Use reStructuredText formatting: `PEP 287 <https://www.python.org/dev/peps/pep-0287/>`_

And for a more Pythonic code: `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_

GIT
...

You can find the development model at http://nvie.com/posts/a-successful-git-branching-model/
Please read this page and stick to it.
The master and develop branches are dedicated to the manager only.
Release and hotfix branches are strongly encouraged. They must be sent to the manager only in a **finished** state.

