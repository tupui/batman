Maintainer / core-developer information
---------------------------------------

Project management
..................

This is an open-source project, thus its development strategy is also open-sourced:

* Project is managed using *gitlab*'s functionalities.
* For a merge request to be integrated, it must be approved at least by one other developer.
  After that, only masters can merge the request.
* Development discussions happen on the `chat <https://batman-cerfacs.zulipchat.com>`_.
  Meeting reports are also posted here.

Making a release
................

Aside from following the `git-flow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model,
here are some additionnal points:

* Write the changelog (``CHANGELOG.rst``). Commit counts can be generated using
  ``git log <last_release>.. | git shortlog -s -n`` for sorting by commits.
* Fix Milestone issues.
* Change the version number in ``batman/__init__.py`` (do not write the *ReleaseName*).
* Update the docs configuration file (credit).
* Compile documentation.
* Ensure that all deprecations have been taken care of.
* Make sure that both python 2 and python 3 `tests <https://gitlab.com/cerfacs/batman/pipelines>`_ pass on master.
* Tag master with X.X-*ReleaseName* (use ``git tag -a`` to annotate the tag).
* Update ``conda`` `recipe <https://github.com/conda-forge/batman-feedstock>`_.
* Share the info on the `chat <https://batman-cerfacs.zulipchat.com>`_.
