# coding: utf8
"""
Setup script for BATMAN
=======================

This script allows to install jpod within the python environment.

Usage
-----
::

    python setup.py build_fortran
    python setup.py install
    python setup.py build_sphinx

"""

import re
import os
import sys
import subprocess
from setuptools import (setup, find_packages, Command)
from distutils.version import LooseVersion

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
        sphinx.build_main(
            ['setup.py', '-b', 'html', './doc', './doc/_build/html'])
        # sphinx.build_main(
        #     ['setup.py', '-b', 'man', './doc', './doc/_build/man'])


class CompileSources(Command):

    """Compile fortran sources."""

    description = 'Compile fortran sources'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('make')
        os.system('make clean')


cmdclasses['build_sphinx'] = BuildSphinx
cmdclasses['build_fortran'] = CompileSources

# Check some import before starting build process.
OPENTURNS_MIN_VERSION = LooseVersion('1.8')
try:
    import openturns
    if LooseVersion(openturns.__version__) < OPENTURNS_MIN_VERSION:
        raise ImportError('Found OpenTurns {}'.format(openturns.__version__))
except ImportError as e:
    msg = '{}{}You need to install OpenTURNS >= {}'
    raise ImportError(msg.format(e, os.linesep, OPENTURNS_MIN_VERSION))

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'mock', 'coverage', 'pylint']
install_requires = ['sphinx_rtd_theme',
                    'sphinx>=1.4',
                    'scipy>=0.15',
                    'numpy>=1.13',
                    'jsonschema',
                    'pathos>=0.2',
                    'matplotlib>=1.5',
                    'scikit-learn>=0.18']

if sys.version_info <= (3, 3):
    install_requires.append('futures')


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
    name='batman',
    version=find_version("batman", "__init__.py"),
    packages=find_packages(exclude=['test_cases', 'doc']),
    entry_points={'console_scripts': ['batman=batman.ui:main']},
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*',
    # Package requirements
    setup_requires=setup_requires,
    tests_require=tests_require,
    install_requires=install_requires,
    extras_require={'Antares': ["antares"]},
    dependency_links=['git+ssh://git@nitrox.cerfacs.fr:cfd-apps/antares.git#egg=Antares-1.9.0'],
    cmdclass=cmdclasses,
    # metadata
    maintainer="Pamphile ROY",
    maintainer_email="roy@cerfacs.fr",
    description="BATMAN creates a surrogate model and perform UQ.",
    long_description=open('./README.rst').read(),
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Environment :: Console',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Natural Language :: English',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.3',
                 'Topic :: Communications :: Email',
                 'Topic :: Documentation :: Sphinx',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 ],
    include_package_data=True,
    zip_safe=False,
    license="CERFACS",
    url=["https://nitrox.cerfacs.fr/open-source/batman",
         "http://open-source.pg.cerfacs.fr/batman"],
)
