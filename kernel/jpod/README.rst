
JPOD
====

- Use POD to build a metamodel for design, optimization and database exchange (loads, MDO, identification)
- Construct model for the full field (conservative variables, local quantities…)
- Optimize the number of CFD computations
- Automatically manage (parallel) the CFD computations and the POD reduction
- Have the optimal representation of the problem for a minimal cost in term of CFD evaluations


Install
-------

You must belong to the ``uqs`` Unix group.

You can get the latest sources with::

    git clone /home/jpod_home/jpod/JPOD
    git clone ssh://dogon.cerfacs.fr/home/jpod_home/jpod/JPOD

Then to install::

    cd kernel
    make


How to use
----------

`JPOD` folder contrains two subfolders: `kernel` and `test_cases`. Adapt a case to you needs.
You can find more information about the cases within the respectives ``README.rst`` file.


Development Model
-----------------

Python
~~~~~~

All developers must follow guidelines from the Python Software Foundation.
As a quick reference:

* For text: "PEP 8": https://www.python.org/dev/peps/pep-0008/
* For documentation: "PEP 257": https://www.python.org/dev/peps/pep-0257/
* Use reStructuredText formatting: "PEP 287": https://www.python.org/dev/peps/pep-0287/

And for a more Pythonic code: "PEP 20": https://www.python.org/dev/peps/pep-0020/

GIT
~~~

You can find the development model at http://nvie.com/posts/a-successful-git-branching-model/
Please read this page and stick to it.
The master and develop branches are dedicated to the manager only.
Release and hotfix branches are strongly encouraged. They must be sent to the manager only in a 'finished' state.
