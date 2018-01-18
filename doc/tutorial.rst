.. _tutorial:


Tutorials
=========

Examples can be found in BATMAN's installer subrepository ``test_cases``. To create a new study called *StudyName*, use the following structure:

.. code::

    StudyName
    |
    ├__ data
    |   |__ script.sh
    |   ├__ function.py
    |
    ├__ settings.json


The working directory consists in two parts: 

+ ``data``: contains all the simulation files necessary to perform a new simulation. It can be a simple python script to a complex code. The content of this directory will be copied for each snapshot. In all cases, ``script.sh`` launches the simulation.

+ ``settings.json``: contains the case setup.

The following sections are a step-by-step tutorial that can be applied to any case and an example of application with an hydraulic modelling software.

.. toctree::
   :maxdepth: 1

   A step-by-step tutorial based on the Michalewicz function <tutorial/michalewicz>
   A detailed application based on a 1D free surface flow model <tutorial/mascaret>
