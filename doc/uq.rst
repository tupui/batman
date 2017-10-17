.. _uq:

Uncertainty Quantification
**************************

What is Uncertainty
===================

As it can be infered from the name, Uncertainty Quantification (UQ) aims at undestanding the impact
of the uncertainties of a system. Uncertainties can be decomposed in two parts:

* Aleatoric: intrinsic variability of a system,
* Epistemic: lack of knowledge, models errors.

The aleatoric part is the one we seek to measure. For example, looking at an airfoil, if we change
the angle of attack, some change are expected on the lift and drag. On the other hand, the epistemic part
represent our bias. Using RANS models, the turbulence is entirelly modeled---as opposed to LES where we compute most of it---so
we might miss some phenomena.

Then, there are three kind of uncrtainty study: 

* Uncertainty Propagation: observe the response of the system to perturbed inputs (PDF, response surface),
* Sensitivity Analysis: measure the respective importance of the input parameters,
* Risk Assessment: get the probability to exceed a threshold.

In any case, from perturbed input we are looking at the output response of a quantity of interest .

.. seealso:: The :ref:`Visualization <visualization>` module is used to output UQ.

*Sobol'* indices
================

There are several methods to estimate the contribution of different parameters on quantities of interest [iooss2015]_.
Among them, sensitivity methods based on the analysis of the variance allow to obtain the contribution of the parameters on the QoI's variance [ferretti2016]_.
Here, classical *Sobol'* [Sobol1993]_ method is used which gives not only a ranking but also quantifies the importance factor using the variance.
This method only makes the hypothesis of the independence of the input variables.
It uses a functional decomposition of the variance of the function to explore:

.. math::
    \mathbb{V}(\mathcal{M}_{gp}) &= \sum_{i}^{p} \mathbb{V}_i (\mathcal{M}_{gp}) + \sum_{i<j}^{p}\mathbb{V}_{ij} + ... + \mathbb{V}_{1,2,...,p},\\
    \mathbb{V}_i(\mathcal{M}_{gp}) &= \mathbb{\mathbb{V}}[\mathbb{E}(\mathcal{M}_{gp}|x_i)]\\
    \mathbb{V}_{ij} &= \mathbb{\mathbb{V}}[\mathbb{E}(\mathcal{M}_{gp}|x_i x_j)] - \mathbb{V}_i - \mathbb{V}_j,

with :math:`p` the number of input parameters constituting :math:`\mathbf{x}`. This way *Sobol'* indices are expressed as

.. math:: S_i = \frac{\mathbb{V}[\mathbb{E}(\mathcal{M}_{gp}|x_i)]}{\mathbb{V}[\mathcal{M}_{gp}]}\qquad S_{ij} = \frac{\mathbb{V}[\mathbb{E}(\mathcal{M}_{gp}|x_i x_j)] - \mathbb{V}_i - \mathbb{V}_j}{\mathbb{V}[\mathcal{M}_{gp}]}.

:math:`S_{i}` corresponds to the first order term which apprises the contribution of the *i-th* parameter,
while :math:`S_{ij}` corresponds to the second order term which informs about the correlations between the *i-th* and the *j-th* parameters.
These equations can be generalized to compute higher order terms.
However, the computational effort to converge them is most often not at hand [iooss2010]_ and their analysis,
interpretations, are not simple.

Total indices represents the global contribution of the parameters on the QoI and express as:

.. math:: S_{T_i} = S_i + \sum_j S_{ij} + \sum_{j,k} S_{ijk} + ... \simeq 1 - S_{i}.

For a functional output, *Sobol'* indices can be computed all along the output and retrieve a map or create composite indices.
As described by Marrel [marrel2015]_, aggregated indices can also be computed as the mean of the indices weighted by the variance at each point or temporal step

.. math:: S_i = \frac{\displaystyle\sum_{l = 1}^{p} \mathbb{V} [\mathbf{f}_l] S_i^{l}}{\displaystyle\sum_{l = 1}^{p} \mathbb{V} [\mathbf{f}_l]}.

The indices are estimated using *Martinez*' formulation. In [baudin2016]_,
they showed that this estimator is stable and provides asymptotic confidence intervals---approximated with Fisher's transformation---for first order and total order indices.


Uncertainty propagation
=======================

Instead of looking at the individual contributions of the input parameters,
the easyest way to assess uncertainties is to perform simulations by perturbing the input distributions
using particular distributions. The quantitie of interest can then be visualized.
This is called a response surface. A complementary analysis can be drawn from here as ones can compute the
Probability Density Function (PDF) of the output. In order for these statistical information to be relevant, a large number of simulations is required.
Hence the need of a surrogate model (see :ref:`Surrogate <surrogate>`).

References
==========

.. [iooss2015] Iooss B. and Saltelli A.: Introduction to Sensitivity Analysis. Handbook of UQ. 2015. DOI: 10.1007/978-3-319-11259-6_31-1 :download:`pdf <ref/Iooss2015.pdf>`
.. [ferretti2016] Ferretti F. and Saltelli A. et al.: Trends in sensitivity analysis practice in the last decade. Science of the Total Environment. 2016. DOI: 10.1016/j.scitotenv.2016.02.133 :download:`pdf <ref/Ferretti2016.pdf>`
.. [Sobol1993] Sobol' I.M. Sensitivity analysis for nonlinear mathematical models. Mathematical Modeling and Computational Experiment. 1993. :download:`pdf <ref/Sobol1993.pdf>`
.. [iooss2010] Iooss B. et al.: Numerical studies of the metamodel fitting and validation processes. International Journal on Advances in Systems and Measurements. 2010 :download:`pdf <ref/Iooss2010.pdf>`
.. [marrel2015] Marrel A. et al.: Sensitivity Analysis of Spatial and/or Temporal Phenomena. Handbook of Uncertainty Quantification. 2015. DOI: 10.1007/978-3-319-11259-6_39-1 :download:`pdf <ref/Marrel2015.pdf>`
.. [baudin2016] Baudin M. et al.: Numerical stability of Sobol’ indices estimation formula. 8th International Conference on Sensitivity Analysis of Model Output. 2016. :download:`pdf <ref/Baudin2016.pdf>`

Sources
=======
.. py:module:: uq
.. automodule:: batman.uq.uq
   :members:
   :undoc-members:
