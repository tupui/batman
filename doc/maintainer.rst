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
* CI is provided by *Gitlab CI* using custom made Docker images. Relative configuration
  and definition files are located under ``.gitlab/continuous_integration``.
  Two images are available on `Docker cloud <https://cloud.docker.com>`_ at:
  ``tupui/bat_ci_2`` and ``tupui/bat_ci_3``.

Making a release
................

Aside from following the `git-flow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model,
here are some additionnal points:

1. Write the changelog (``CHANGELOG.rst``). Commit counts can be generated using
   ``git log <last_release>.. | git shortlog -s -n`` for sorting by commits.
2. Fix Milestone issues.
3. Change the version number in ``batman/__init__.py`` (do not write the *ReleaseName*).
4. Update the docs configuration file (credit).
5. Compile documentation.
6. Ensure that all deprecations have been taken care of.
7. Update Docker images and upload the python 3 image on Gitlab registry::
   
    docker build -t tupui/bat_ci_[2,3] -f Dockerfile_python_[2,3] .

    docker login -u tupui -p xxx
    docker push tupui/bat_ci_[2,3]

    docker login registry.gitlab.com -u tupui -p xxx
    docker tag tupui/bat_ci_3 registry.gitlab.com/cerfacs/batman/tupui/bat_ci_3
    docker push registry.gitlab.com/cerfacs/batman/tupui/bat_ci_3

9. Make sure that both python 2 and python 3 `tests <https://gitlab.com/cerfacs/batman/pipelines>`_ pass on master.
10. Tag master with X.X-*ReleaseName* (use ``git tag -a`` to annotate the tag).
11. Update ``conda`` `recipe <https://github.com/conda-forge/batman-feedstock>`_ (version and commit patch).
12. Update ``pip`` package::

    python setup.py sdist bdist_wheel
    twine upload dist/*

12. Share the info on the `chat <https://batman-cerfacs.zulipchat.com>`_.
