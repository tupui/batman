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

try:
    import scipy
except ImportError:
    import pip
    pip.main(['install', 'scipy'])

sphinx_requires = ['sphinx', 'sphinx_rtd_theme']

setup(
    name='jpod',
    version='1.2.dev0',
    packages=find_packages(),
    entry_points={'console_scripts': ['jpod=jpod.ui:main']},
    # Package requirements
    install_requires=['sphinx_rtd_theme', 'futures', 'rpyc', 'mpi4py', 'h5py', 'scikit-learn>=0.18.dev0'],
    dependency_links=['https://github.com/scikit-learn/scikit-learn/archive/master.zip#egg=scikit-learn-0.18.dev0'],
    cmdclass=cmdclasses,
    # metadata
    maintainer="Pamphile ROY",
    maintainer_email="roy@cerfacs.fr",
    description="JPOD creates a surrogate model using POD+Kriging and perform UQ.",
    long_description=open('./doc/README.rst').read(),
    classifiers=['Development Status :: 3 - Alpha',
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
    license="CERFACS",
    url="https://inle.cerfacs.fr/projects/jpod",
)
