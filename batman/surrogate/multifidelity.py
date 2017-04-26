# coding: utf8
"""
Evofusion Class
===============

Interpolation using Evofusion method.Evofusion


Reference
---------

Forrester, Sobester et al.: Multi-Fidelity Optimization via Surrogate Modelling. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences. 2007. DOI 10.1098/rspa.2007.1900

"""
import numpy as np
import logging


class Evofusion(object):

    """Multifidelity algorithm using Evofusion."""

    logger = logging.getLogger(__name__)

    def __init__(self, input, output):
        pass
