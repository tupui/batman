---
title: 'BATMAN: Bayesian Analysis Tool for Modelling and uncertAinty
quaNtification'
tags:
  - Python
  - Uncertainty Quantification
  - Statistical analysis
  - Surrogate Model
  - Design of Experiments
  - Computational Fluid Dynamics
authors:
 - name: Pamphile T. Roy
   orcid: 0000-0001-9816-1416
   affiliation: 1
 - name: Sophie Ricci
   affiliation: 1
 - name: Romain Dupuis
   affiliation: 2
 - name: Robain Campet
   affiliation: 1
 - name: Jean-Christophe Jouhaud
   affiliation: 1
 - name: Cyril Fournier
   affiliation: 1
affiliations:
 - name: CERFACS, Toulouse, France
   index: 1
 - name: IRT/CERFACS, Toulouse, France
   index: 2
date: 12 November 2017
bibliography: paper.bib
---

# Summary

Numerical codes have reached a sufficient maturity to represent physical phenomena, and complex simulations on high-resolution grid is becoming possible with continuous developments in numerical methods and in High Performance Computing (HPC). Still, deterministic simulations only provide limited knowledge on a system as uncertainties in the numerical model and its inputs translate into uncertainties in the outputs.

Batman is a python module that allows to seamlessly perform statistical analysis using any experiment. From a simple setting file, it handles all necessary steps automatically. Another possibility is to access batman API to independently use all functionalities. Last but not least, batman can make use of existing results from both in vivo and in silico experimentations.

Along with state-of-the-art algorithms for creating design of experiments (DoE), constructing a surrogate model (SM) and performing uncertainty quantification (UQ), it implements new methods for resampling the parameter space and new data visualization methods to assess uncertainties [@roy2017a]. Batman has been successfully used to treat complex Computational Fluid Dynamics applications [@roy2017a, @roy2017b, @roy2017c].

# References
