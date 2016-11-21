# coding: utf8
"""
Setup script for JPOD
=====================

This script allows to install jpod within the python environment.

Usage
-----
::

    python setup.py build_fortran
    python setup.py install
    python setup.py build_sphinx

"""

from setuptools import (setup, find_packages, Command)
import re
import os
import subprocess

cmdclasses = dict()


class BuildSphinx(Command):

    """Build Sphinx documentation."""

    description = 'Build Sphinx documentation'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sphinx
        sphinx.build_main(['setup.py', '-b', 'html', './doc', './doc/_build/html'])
        sphinx.build_main(['setup.py', '-b', 'man', './doc', './doc/_build/man'])


class CompileSources(Command):

    """Compile fortran sources."""

    description = 'Compile fortran sources'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system('make')
        os.system('make clean')


cmdclasses['build_sphinx'] = BuildSphinx
cmdclasses['build_fortran'] = CompileSources

# Check some import before starting build process.
try:
    import scipy
except ImportError:
    import pip
    pip.main(['install', 'scipy'])

try:
    import mpi4py
except ImportError:
    try:
        import pip
        pip.main(['install', 'mpi4py'])
    except:
        print('You need to have a proper MPI installation')
        raise SystemExit

try:
    import openturns
except:
    print('You need to install OpenTURNS')
    raise SystemExit


def find_version(*file_paths):
    """Find version number, commit and branch."""
    with open(os.path.join(os.path.dirname(__file__), *file_paths),
              'r') as f:
        version_file = f.read()
    commit = subprocess.check_output("git describe --always",
                                     shell=True).rstrip()
    branch = subprocess.check_output("git describe --all",
                                     shell=True).rstrip()
    version_file = re.sub('(__commit__ = )(.*)',
                          r'\g<1>' + "'" + commit.decode('utf8') + "'",
                          version_file)
    version_file = re.sub('(__branch__ = )(.*)',
                          r'\g<1>' + "'" + branch.decode('utf8') + "'",
                          version_file)

    with open(os.path.join(os.path.dirname(__file__), *file_paths),
              'wb') as f:
        f.write(version_file.encode('utf8'))
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='jpod',
    version=find_version("jpod", "__init__.py"),
    packages=find_packages(exclude=['test_cases', 'doc']),
    entry_points={'console_scripts': ['jpod=jpod.ui:main']},
    # Package requirements
    install_requires=['sphinx_rtd_theme',
                      'sphinx',
                      'jsonschema',
                      'futures',
                      'pathos',
                      'otwrapy==0.6',
                      'rpyc',
                      'h5py',
                      'scikit-learn>=0.18'],
    dependency_links=['https://github.com/felipeam86/otwrapy/tarball/master#egg=otwrapy-0.6'],
    cmdclass=cmdclasses,
    # metadata
    maintainer="Pamphile ROY",
    maintainer_email="roy@cerfacs.fr",
    description="JPOD creates a surrogate model using \
        POD+Kriging and perform UQ.",
    long_description=open('./README.rst').read(),
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Environment :: Console',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Natural Language :: English',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Topic :: Communications :: Email',
                 'Topic :: Documentation :: Sphinx',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 ],
    include_package_data=True,
    zip_safe= False,
    license="CERFACS",
    url="https://inle.cerfacs.fr/projects/jpod",
)
