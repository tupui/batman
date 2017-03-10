.. _space:

Sampling the Space of Parameters
================================

Design Of Experiments
---------------------

Whatever method is used, the first step consists in defining how we are going to modify input variables to retrieve the evolution of the response surface. This is called a Design Of Experiments (DOE) as defined by [Sacks1989]_. 

.. image:: ./fig/monteCarlo.pdf

The quality of the DOE is paramount as it determines the physics that will be observed. If the space is not filled properly, homogeneously, we can bias our analysis and retrieve only a particular behaviour of the physic. This concept has been extensively been used in experiments, especially the one-at-a-time design, which consists of only changing only one parameter at a time. Doing so, the space is not filled properly and only simple behaviours can be recovered. In order to assess the quality of the sampling, the discrepancy is usually used. It is an indicator of the distance between the points within the space of parameters. The lower the discrepancy is, the better the design is.

As stated before, the golden standard would be to perform a *Monte Carlo* sampling but it would require a huge sampling which is unfeasible with costly numerical simulations. Therefore are found random (or quasi-random) sampling methods. Low discrepancy sequence has been designed to overcome this issue. These designs are built upon a pattern, a sequence, depending on factors such as prime numbers. This allows a fast generation of sampling space with good properties. A well-known method is the Latin Hypercube Sampling (LHS). The idea behind is to discretize the space to get a regular grid and sample randomly a point per zone.

In Damblin et al. [Damblin2013]_ a comprehensive analysis of most common DOE is found. In the end, the *Sobol'* or *Halton* DOE are sufficient when dealing with a small number of parameters (<5). With an increasing number of parameters, patterns start to appear.

References
..........

.. [Damblin2013] G. Damblin, M. Couplet, B. Iooss: Numerical studies of space filling designs : optimization of Latin Hypercube Samples and subprojection properties. Journal of Simulation. 2013.

.. [Sacks1989] J. Sacks et al.: “Design and Analysis of Computer Experiments”. Statistical Science 4.4. 1989. DOI: 10.1214/ss/1177012413

Space module
------------

.. automodule:: batman.space.space
   :members:
   :undoc-members:

.. automodule:: batman.space.sampling
   :members:
   :undoc-members:

.. automodule:: batman.space.point
   :members:
   :undoc-members:

.. automodule:: batman.space.refiner
   :members:
   :undoc-members:

