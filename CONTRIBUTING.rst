.. _contributing:

Developer Guide
===============

Introduction
------------

First of all, if you are reading this, thank you and welcome to this community. For the efficiency and the serenity of all users, every good python projects requires some guidelines to be observed. Every code seeks to be competitive, usable, stable and maintainable. This can only be achieve through a high test coverage, a good documentation and coding consistency. So, if you wish to contribute, you must comply to the following rules for your pull request to be considered.


Respect Python features
-----------------------

As this is a python project, you have to adapt your thinking to the python style. Otherwise, this can lead to performance issues.
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

Spot basic errors
-----------------

Appart from normal unit and integration tests, you can perform a static
analysis of the code using `pylint <https://www.pylint.org>`_::

    pylint batman --rcfile=setup.cfg --ignore-patterns='gp_1d_sampler.py','RBFnet.py','TreeCut.py','resampling.py'

This allows to spot naming errors for example as well as style errors.

Running developments on existing test-cases 
-------------------------------------------

Testing your code is paramount. Without continuous integration, you **cannot**
guaranty the quality of the code. Some minor modification on a module can have
unexpected implications. With a single commit, everything can go south!
The ``master`` branch, and normally the ``develop`` branch, are always on a
passing state. This means that you should be able to checkout from them an use
BATMAN without any errors.

The library `pytest <https://docs.pytest.org/en/latest/>`_ is used. It is simple and powerfull.
Checkout their doc and replicate constructs from existing tests. If you are not
already in love with it, you will soon be. All tests can be launched using::

    coverage run -m pytest --basetemp=./TMP_CI batman/tests test_cases

This command fires `coverage <http://coverage.readthedocs.io>`_ at the same time.
The output consists in tests results and coverage report.

.. note:: Tests will be automatically launched when you will push your branch to
  the server. So you only have to run locally your new tests or the one you
  think you should.

Working under GIT
-----------------

The ``master`` and ``develop`` branches are protected and dedicated to the manager only.
Release and hotfix branches are mandatory.

If you want to add a modification, create a new branch branching off ``develop``.
Then you can create a merge request on *gitlab*. From here, the fun beggins.
You can commit any change you feel, start discussions about it, etc.

1. Clone this copy to your local disk::

        $ git clone git@nitrox.cerfacs.fr:open-source/batman.git

2. Create a branch to hold your changes::

        $ git checkout -b my-feature

   and start making changes. Never work in the ``master`` branch!

3. Work on this copy, on your computer, using Git to do the version
   control. When you're done editing, do::

        $ git add modified_files
        $ git commit

   to record your changes in Git, then push them to GitHub with::

        $ git push -u origin my-feature

4. Finally, follow `these <https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html>`_
   instructions to create a merge request from your fork. This will send an
   email to the committers.

.. note:: For every commit you push, the linter is launched. After that, if you
  want to launch all tests, you have to manually run them using the interface button.

Your request will only be considered for integration if in a **finished** state: 

0. Respect python coding rules,
1. Maintain linting score (>9.5/10), 
2. The branch passes all tests,
3. Have tests regarding the changes,
4. Maintain test coverage,
5. Have the respective documentation.

.. note:: You can find the development model at http://nvie.com/posts/a-successful-git-branching-model/

Submit your developments
------------------------

`Here <https://cerfacs.gitlab.io/batman/contributing_link.html>`_


API Reference
-------------

This is the class and function reference of batman. Please refer to
previous sections for further details, as the class and function raw
specifications may not be enough to give full guidelines on their uses.

:mod:`batman.space`: Parameter space
....................................

.. .. automodule:: batman.space
..    :no-members:
..    :no-inherited-members:

.. currentmodule:: batman

.. autosummary::

   space.Point
   space.Space
   space.Doe
   space.Refiner

.. py:module:: space
.. automodule:: batman.space
   :members:
   :undoc-members:

:mod:`batman.surrogate`: Surrogate Modelling
............................................

.. currentmodule:: batman

.. autosummary::

   surrogate.SurrogateModel
   surrogate.Kriging
   surrogate.PC
   surrogate.RBFnet

.. py:module:: surrogate
.. automodule:: batman.surrogate
   :members:
   :undoc-members:

:mod:`batman.uq`: Uncertainty Quantification
............................................

.. currentmodule:: batman

.. autosummary::

   uq.UQ

.. py:module:: uq
.. automodule:: batman.uq
   :members:
   :undoc-members:

:mod:`batman.visualization`: Uncertainty Visualization
......................................................

.. currentmodule:: batman

.. autosummary::

   visualization.Kiviat3D
   visualization.HdrBoxplot
   visualization.doe
   visualization.response_surface
   visualization.sobol
   visualization.corr_cov
   visualization.pdf
   visualization.kernel_smoothing
   visualization.reshow

.. py:module:: visualization
.. automodule:: batman.visualization
   :members:
   :undoc-members:

:mod:`batman.pod`: Proper Orthogonal Decomposition
..................................................

.. currentmodule:: batman

.. autosummary::

   pod.Pod

.. py:module:: pod
.. automodule:: batman.pod
   :members:
   :undoc-members:

:mod:`batman.functions`: Functions
..................................

.. currentmodule:: batman

.. autosummary::

   functions.data
   functions.analytical.SixHumpCamel
   functions.analytical.Branin
   functions.analytical.Michalewicz
   functions.analytical.Ishigami
   functions.analytical.Rastrigin
   functions.analytical.G_Function
   functions.analytical.Forrester
   functions.analytical.ChemicalSpill
   functions.analytical.Channel_Flow
   functions.analytical.Manning
   functions.telemac_mascaret.Mascaret
   functions.telemac_mascaret.MascaretApi
   functions.utils.multi_eval
   functions.utils.output_to_sequence

.. py:module:: functions
.. automodule:: batman.functions.data
   :members:
   :undoc-members:

.. automodule:: batman.functions.analytical
   :members:
   :undoc-members:

.. automodule:: batman.functions.telemac_mascaret
   :members:
   :undoc-members:

:mod:`batman.tasks`: Tasks
..........................

.. currentmodule:: batman

.. autosummary::

   tasks.SnapshotTask
   tasks.SnapshotProvider
   tasks.Snapshot

.. py:module:: tasks
.. automodule:: batman.tasks
   :members:
   :undoc-members:

:mod:`batman.misc`: Misc
........................

.. currentmodule:: batman

.. autosummary::

   misc.NestedPool
   misc.ProgressBar
   misc.optimization
   misc.import_config
   misc.check_yes_no
   misc.ask_path
   misc.abs_path
   misc.clean_path

.. py:module:: misc
.. automodule:: batman.misc
   :members:
   :undoc-members:

:mod:`batman.input_output`: Input Output
........................................

.. currentmodule:: batman

.. autosummary::

   input_output.Dataset
   input_output.IOFormatSelector

.. py:module:: input_output
.. automodule:: batman.input_output
   :members:
   :undoc-members:


.. [Wand1995] M.P. Wand and M.C. Jones. Kernel Smoothing. 1995. DOI: 10.1007/978-1-4899-4493-1 
.. [Roy2017b] P.T. Roy et al.: Comparison of Polynomial Chaos and Gaussian Process surrogates for uncertainty quantification and correlation estimation of spatially distributed open-channel steady flows. SERRA. 2017. DOI: 10.1007/s00477-017-1470-4 

