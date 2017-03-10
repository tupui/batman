.. _contributing:

Contributing
============

If you wish to contribute to this project, you **must** follow the following
for your pull request to be considered.

Install
-------

The procedure is similar to the end-user one but if you plan to modify the
sources, you need to install it with::

    python setup.py develop

This will create a simlink to your python install folder. Thus you won't have
to re-install the package after you modified it.

Python
------

All developers must follow guidelines from the Python Software Foundation. As a quick reference:

* For text: `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
* For documentation: `PEP 257 <https://www.python.org/dev/peps/pep-0257/>`_
* Use reStructuredText formatting: `PEP 287 <https://www.python.org/dev/peps/pep-0287/>`_

And for a more Pythonic code: `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_
Last but not least, avoid common pitfalls: `Anti-patterns <http://docs.quantifiedcode.com/python-code-patterns/>`_

GIT
---

You can find the development model at http://nvie.com/posts/a-successful-git-branching-model/
Please **read** this page and **stick** to it.
The master and develop branches are dedicated to the manager only.
Release and hotfix branches are mandatory. They must be sent to the manager only in a **finished** state.
