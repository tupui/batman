.. _contributing:

Contributing
============

If you are reading this, first of all, thank you and welcome to this community.
For everyone to have fun, every good python projects requires some guidelines
to be observed.

Every code seek to be performant, usable, stable and maintainable.
This can only be acheave through high test coverage, good documentation and
coding consistency. Isn't it frustrating when you cannot understand some code
just because there is no documentation nor any test to assess that the function
is working nor any comments in the code itself? How are you supposed to code in
these conditions?

If you wish to contribute, you **must** comply to the following rules for your
pull request to be considered.

Install
-------

The procedure is similar to the end-user one but if you plan to modify the
sources, you need to install it with::

    python setup.py develop

This will create a simlink to your python install folder. Thus you won't have
to re-install the package after you modified it.

Python
------

This is a python project, not some C or Fortran code. You have to adapt your
thinking to the python style. Otherwise, this can lead to performance issues.
For example, an ``if`` is expensive, you would be better off using a ``try except``
construction. *It is better to ask forgiveness than permission*. Also, when
performing computations, care to be taken with ``for`` loops. If you can, use
*numpy* operations for huge performance impact (sometimes x1000!).

Thus developers **must** follow guidelines from the Python Software Foundation.
As a quick reference:

* For text: `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
* For documentation: `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_
* Use reStructuredText formatting: `PEP 287 <https://www.python.org/dev/peps/pep-0287/>`_

And for a more Pythonic code: `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_
Last but not least, avoid common pitfalls: `Anti-patterns <http://docs.quantifiedcode.com/python-code-patterns/>`_

Linter
------

Appart from normal unit and integration tests, you can perform a static
analysis of the code using `pylint <https://www.pylint.org>`_::

    pylint batman --rcfile=setup.cfg --ignore-patterns='gp_1d_sampler.py','RBFnet.py','TreeCut.py','resampling.py'

This allows to spot naming errors for example as well as style errors.

Testing
-------

Testing your code is paramount. Without continuous integration, you **cannot**
guaranty the quality of the code. Some minor modification on a module can have
unexpected implications. With a single commit, everything can go south!
The ``master`` branch, and normally the ``develop`` branch, are always on a
passing state. This means that you should be able to checkout from them an use
BATMAN without any errors.

The library `pytest <https://docs.pytest.org/en/latest/>`_ is used. It is simple and powerfull.
Checkout their doc and replicate constructs from existing tests. If you are note
already in love with it, you will soon be. All tests can be launched using::

    coverage run -m pytest --basetemp=./TMP_CI .
    coverage report -m

These commands fire `coverage <http://coverage.readthedocs.io>`_ in the same time.
The output consists in tests results and coverage report.

GIT
---

You can find the development model at http://nvie.com/posts/a-successful-git-branching-model/
Please **read** this page and **stick** to it.
The ``master`` and ``develop`` branches are protected and dedicated to the manager only.
Release and hotfix branches are mandatory.

If you want to add a modification, create a new branch branching off ``develop``.
Then you can create a merge request on *gitlab*. From here, the fun beggins.
You can commit any change you feel, start discussions about it, etc.

Your request will only be considered for integration if in a **finished** state: 

0. Respect python coding rules,
1. Maintain linting score (>9.5/10), 
2. The branch passes all tests,
3. Have tests regarding the changes,
4. Maintain test coverage,
5. Have the respective documentation.
