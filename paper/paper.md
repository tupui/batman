---
title: 'BATMAN: Statistical analysis for expensive computer codes made easy'
tags:
  - Python
  - Uncertainty Quantification
  - Statistical Analysis
  - Surrogate Model
  - Design of Experiments
  - Uncertainty Visualization
authors:
 - name: Pamphile T. Roy
   orcid: 0000-0001-9816-1416
   affiliation: 1
 - name: Sophie Ricci
   orcid: 0000-0002-4232-5626
   affiliation: 1
 - name: Romain Dupuis
   affiliation: 2
 - name: Robin Campet
   orcid: 0000-0002-4434-0854
   affiliation: 1
 - name: Jean-Christophe Jouhaud
   affiliation: 1
 - name: Cyril Fournier
   affiliation: 1
affiliations:
 - name: CERFACS, Toulouse, France
   index: 1
 - name: IRT Saint Exupéry/CERFACS, Toulouse, France
   index: 2
date: 23 November 2017
bibliography: paper.bib
---

# Summary

Bayesian Analysis Tool for Modelling and uncertAinty
quaNtification (batman) is an open source Python package dedicated to statistical analysis based on non-intrusive ensemble experiment.

Numerical software has reached a sufficient maturity to represent physical phenomena. High fidelity simulation is possible with continuous advances in numerical methods and in High Performance Computing (HPC). Still, deterministic simulations only provide limited knowledge on a system as uncertainties in the numerical model and its inputs translate into uncertainties in the outputs. Ensemble-based methods are used to construct a numerical or experimental dataset from which statistics are inferred.

*batman* library provides a convenient, modular and efficient framework for design of experiments, surrogate model and uncertainty quantification. *batman* relies on open source python packages dedicated to statistics (*openTURNS* and *scikit-learn* [@openturns, @scikit-learn]). It also implements advanced methods for resampling, robust optimization and uncertainty visualization [@roy2017a].

*batman* handles the workflow for statistical analysis. It makes the most of HPC resources by managing asynchronous parallel tasks. The internal parallelism of each task does not conflict with *batman*'s parallel environment.

*batman* analysis is launched from a *command line interface* and a setting file. *batman* functionalities can also be accessed through an API. *batman* has been successfully used for geosciences and turbomachinery Computational Fluid Dynamics applications [@roy2017a, @roy2017b, @roy2017c].

*batman* is CECILL-B licensed; it is actively developed and maintained by researchers at CERFACS.

# References

