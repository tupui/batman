.. _space:
.. currentmodule:: batman.space

Sampling the Space of Parameters
================================


Design of Experiments
---------------------

Whatever method is used, the first step consists in defining how we are going to modify input variables to retrieve the evolution of the response surface. This is called a Design of Experiments (DoE) as defined by [Sacks1989]_. The parameter space is called a ``Space``::

    space = batman.space.Space([[1, 1], [3, 3]])
    space.sampling(10, 'halton')
    space.write('.')

.. image:: ../fig/halton_25.pdf

The quality of the DoE is paramount as it determines the physics that will be observed. If the space is not filled properly, homogeneously, we can bias our analysis and retrieve only a particular behaviour of the physic. This concept has been extensively been used in experiments, especially the one-at-a-time design, which consists of only changing only one parameter at a time. Doing so, the space is not filled properly and only simple behaviours can be recovered. In order to assess the quality of the sampling, the discrepancy is usually used. It is an indicator of the distance between the points within the parameters space. The lower the discrepancy is, the better the design is. This information can be used to optimize a DoE Among all formulations of this criterion, the centered discrepancy is the most robust one [Damblin2013]_.
This information can be computed from the space::

    space.discrepancy()

As stated before, the golden standard would be to perform a *Monte Carlo* sampling but it would require a huge sampling which is unfeasible with costly numerical simulations. Therefore are found random (or quasi-random) sampling methods. Low discrepancy sequence has been designed to overcome this issue. These designs are built upon a pattern, a sequence, depending on factors such as prime numbers. This allows a fast generation of sampling space with good properties. A well-known method is the Latin Hypercube Sampling (LHS). The idea behind is to discretize the space to get a regular grid and sample randomly a point per zone.

In Damblin et al. [Damblin2013]_ a comprehensive analysis of most common DOE is found. In the end, the *Sobol'* or *Halton* DOE are sufficient when dealing with a small number of parameters (<5). With an increasing number of parameters, patterns start to appear and optimized LHS are required.

Parameters whose distributions are Gaussian samplers are treated independently of the other variables, using the class ``space.gp_sampler.GpSampler``.

Gaussian process sampler
------------------------

Let :math:`x(t)` be a real parameter indexed by :math:`t\in\mathbf{R}^d` with :math:`d\in\{1,2,3\}`. :math:`t` can represent a time point (:math:`d=1`), a space point (:math:`d=1` or :math:`d=2`) or a time-space point (:math:`d=1`, :math:`d=2` or :math:`d=3`).

This parameter is observed at many index values:
  
    reference = {'indices': [[t_11, ..., t_1d], ..., [t_n1, ..., t_nd]], 'values': [v_1, ..., v_n]}
    
and this parameter ``reference`` can be either an object of type ``list`` or a .npy file storing this object.
    
A Gaussian process with zero mean is built from ``reference`` and a kernel function. The Gaussian process information can be compressed by thresholding the Karhunen Loeve Decomposition (KLD) involved in the Gaussian process sampling. This threshold corresponds to the percentage of Gaussian process variance explained by the truncation of the KLD.

Lastly, Gaussian process realizations can either be added to the reference values [v_1, ..., v_n] and the totals become inputs of the output provider, or directly become inputs of the ouput provider.

Resampling the parameters space
-------------------------------

There are several methods for refining, resampling, the parameter space. In [Scheidt]_, the classical methods are reviewed and a framework combining several methods is proposed. In [Roy2017]_, we added some methods that peforme better in high dimentionnal cases.
In BATMAN, the following methods are available:

* Variance (:math:`\sigma`),
  As stated in :ref:`Surrogate <surrogate>`, one of the main advantages of Gaussian processes over other surrogates is to provide an insight into the variance of the solution. The first method consists in using this data and weight it with the eigenvalues of the POD:

  .. math:: \sum_{i=1}^k \sigma_i^2 \times \mathbb{V}[f(\mathbf{x}_*)]_{i}.

  Global optimization on this indicator gives the new point to simulate.

* Leave-One-Out (LOO) and :math:`\sigma`,
  A LOO is performed on the POD and highlights the point where the model is the most sensitive. The strategy here is to add a new point around it. Within this hypercube, a global optimization over :math:`\sigma` is conduced giving the new point.

* LOO-*Sobol'*,
  Using the same steps as with the LOO - :math:`\sigma` method, the hypercube around the point is here truncated using prior information about *Sobol'* indices-see :ref:`UQ <uq>`. It requires that indices be close to convergence not to bias the result. Or the bias can be intentional depending on the insight we have about the case.

* Extrema,
  This method will add 4 points. First, it look for the point in the sample which has the min value of the QoI. Within an hypercube, it add the minimal and maximal predicted values. Then it do the same for the point of the sample which has the max value of the QoI. This method allows to capture the gradient around extrem values.

* Hybrid,
  This last method consists of a navigator composed by any combination of the previous methods.

* Discrepancy.
  Simply add a point that minimize the discrepancy.


It is fairly easy to resample the parameter space. From a space and a surrogate::

    new_point = space.refine(surrogate)

Hypercube
.........

The hypercube is defined by the cartesian product of the intervals of the :math:`n` parameters *i.e.* :math:`[a_i, b_i]^n`. The constrained optimization problem can hence be written as:

.. math::
   \left\{\begin{array}{rc} \max  &\parallel (\mathbf{b} - \mathbf{a}) \parallel_{2} \\\mathcal{P} &\notin [a_i, b_i]^n \\ p &\in [a_i, b_i]^n \end{array}\right. .

Moreover, a maximum cube-volume aspect ratio is defined in order to preserve the locality. This gives the new constrain

.. math::
   C : \sqrt[n]{\frac{\max (\mathbf{b} - \mathbf{a})}{\displaystyle\prod_{i = 1}^n \max (b_i - a_i)}} < \epsilon ,

with :math:`\epsilon = 1.5` is set arbitrarily to prevent too elongated hypercubes. The global optimum is found using a two-step strategy: first, a discrete optimization using :math:`\mathcal{P}` gives an initial solution; second a basin-hopping algorithm finds the optimum coordinates of the hypercube. In case of the LOO-*Sobol'* method, the hypercube is truncated using the total order *Sobol'* indices.


Efficient Global Optimization (EGO)
-----------------------------------

In the case of a surrogate model based on a gaussian process, Efficient Global Optimization (EGO) [Jones1998]_ algorithm can be used to resample the parameter space in directive to an optimization. It comprises as a tradeoff between the actual minimal value :math:`f_{min}` and an expected value given by the standard error :math:`s` for a given prediction :math:`\hat{y}`. The expected improvement is defined as:

.. math:: \mathbb{E}[I(x)] = (f_{min} - \hat{y})\Phi \left( \frac{f_{min} - \hat{y}}{s} \right) + s\phi \left( \frac{f_{min} - \hat{y}}{s} \right),

with :math:`\phi(.), \Phi(.)` the standard normal density and distribution function. Using the fact that this quantify is monotonic in :math:`\hat{y}` and :math:`s`, it reduces to the probability of improvement:

.. math:: \frac{\partial \mathbb{E}[I(x)]}{\partial \hat{y}} &= - \Phi \left( \frac{f_{min} - \hat{y}}{s} \right) < 0, \\
          \frac{\partial \mathbb{E}[I(x)]}{\partial s} &= \phi \left( \frac{f_{min} - \hat{y}}{s} \right) > 0.
