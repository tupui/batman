.. _surrogate:
.. currentmodule:: batman.surrogate

Surrogate model
===============

Generalities
------------

A common class is used to manage surrogate models. Hence, several kind of surrogate model strategies can be used::

    predictor = batman.surrogate.SurrogateModel('kriging', corners, max_points_nb)
    predictor.fit(space, target_space)
    predictor.save('.')
    points = [(12.5, 56.8), (2.2, 5.3)]
    predictions = predictor(points)

From *Kriging* to *Gaussian Process*
------------------------------------

Kriging, a geostatistical method
................................

*Kriging* is a geostatistical interpolation method that use not only the distance between the neighbouring points but also the relationships among these points, the autocorrelation. The method has been created by D.G. Krige [Krige1989]_ and has been formalized by G. Matheron [Matheron1963]_.

In order to predict an unmeasured location :math:`\hat{Y}`, interpolation methods use the surrounding measured values :math:`Y_i` and weight them:

.. math::
    \hat{Y} = \sum_{i = 1}^{N} \lambda_i Y_i.

The advantage of this method is that the interpolation is exact at the sampled points and that it gives an estimation of the prediction error. Ordinary *Kriging* consists in the *Best Linear Unbiased Predictor* (BLUP) [Robinson1991]_:

Best
    It minimizes the variance of the predicted error :math:`Var(\hat{Y} - Y)`,

Linear
    A linear combination of the data,

Unbiased
    It minimizes the mean square error :math:`E[\hat{Y} - Y]^2` thus :math:`\sum_{i=1}^{N} \lambda_i(x)=1`, 

Predictor
    It is an estimator of random effects.

:math:`\lambda_i` are calculated using the spatial autocorrelation of the data, it is a variography analysis. Plots can be constructed using semivariance, covariance or correlation. An empirical variogram plot allows to see the values that should be alike because they are close to each other \cite{Bohling2005}. The empirical semivariogram is given by:

.. math::
    \gamma(h) = \frac{1}{2}\times \frac{1}{n} \sum_{i=1}^{N} (Y_i - Y_{i+h})^2.

A fitting model is then applied to this semivariogram. Hence, the variability of the model is inferior to data's. Kriging smooths the gradients. The exponential model is written as:

.. math::
    \gamma(h) = C(0) + C\left(1- \exp{\left(-\frac{h}{r}\right)}\right),

with :math:`C` the correlation matrice and the parameter :math:`r` is optimized using the sample points.

.. image:: ../fig/semivariogramme.pdf

A model is described using:

Sill
    It corresponds to the maximum of :math:`\gamma`. It defines the end of the range.

Range
    It is the zone of correlation. If the distance is superior to the range, there is no correlation, whereas if the distance is inferior to it, the sample locations are autocorrelated.

Nugget
    If the distance between the points is null, :math:`\gamma` should be null. However, measurement errors are inherent and cause a nugget effect. It is the y-intercept of the model.

Once the model is computed, the weights are determined to use the *MSE* condition and gives:

.. math:: \lambda_i = K^{-1}k,
 
:math:`K` being the covariance matrix :math:`K_{i,j} = C(Y_i-Y_j)` and :math:`k` being the covariance vector :math:`k_i = C(Y_i-Y)` with the covariance :math:`C(h) = C(0) - \gamma(h) = Sill-\gamma(h)`.
 
.. math::
    \begin{pmatrix}\gamma_{11}& \cdots & \gamma_{1j} \\ \vdots & \ddots & \vdots \\ \gamma_{i1} & \cdots  & \gamma_{nn}  \end{pmatrix} \begin{pmatrix}\lambda_1 \\ \vdots \\ \lambda_n \end{pmatrix} = \begin{pmatrix} \gamma_{1X} \\ \vdots \\ \gamma_{nX}\end{pmatrix}.

Furthermore we can express the field :math:`Y` as :math:`\hat{Y} = R(S) + m(S)` which is the residual and the trend components [Bohling2005]_. Depending on the treatment of the trend, there are several Kriging techniques (ordinary Kriging being the most used):

Simple
    The variable is stationary, the mean is known,

Ordinary
    The variable is stationary, the mean is unknown,

Universal
    The variable is non-stationary, there is a tendency.

Ordinary Kriging is the most used method. In this case, the covariance matrix is augmented:

.. math::
    \begin{pmatrix}\gamma_{11}& \cdots  & \gamma_{1j} & 1\\ \vdots & \ddots & \vdots & \vdots \\ \gamma_{i1} & \cdots  & \gamma_{nn} & 1 \\ 1 & \cdots & 1 & 0 \end{pmatrix} \begin{pmatrix}\lambda_1 \\ \vdots \\ \lambda_n \\ - \mu \end{pmatrix} = \begin{pmatrix} \gamma_{1X} \\ \vdots \\ \gamma_{nX} \\ 1\end{pmatrix}.

Once the weights are computed, its dot product with the residual :math:`R_i=Y_i-m` at the known points gives the residual :math:`R(S)`. Thus we have an estimation of :math:`\hat{Y}`. Finally, the error is estimated by the second order moment:

.. math:: \sigma^2 =  \sum_{i = 1}^{N} \lambda_i \gamma_{iX} - \mu.

Some care has to be taken with this estimation of the variance. Being a good indicator of the correctness of the estimation, this is only an estimation of the error based upon all surrounding points.

Gaussian Process
................

There are two approaches when dealing with regression problems. In simple cases, we can use simple functions in order to approximate the output set of data. On the other hand, when dealing with complex multidimensional problems with strong non-linearity, there are infinite possibilities of functions to consider. This is where the Gaussian process comes in.

As stated by Rasmussen et al. in [Rasmussen2006]_, a process is a generalization of a probability distribution of functions. When dealing with *Gaussian processes*, they can simply be fully defined using the mean and covariance of the functions: 

.. math::
    f(x)&\sim GP(m(x), k(x,x')),\\
    m(x) &= \mathbb{E}\left[ f(x)  \right], \\
    k(x,x') &= \mathbb{E}\left[ (f(x) -m(x))(f(x')-m(x')) \right].

.. figure:: ../fig/rasmussenGP.png

    Subfigure (a) shows four samples from a prior distribution. (b) shows the situation after two observations have been made. [Rasmussen2006]_.

Starting from a prior distribution of functions, it represents the belief we have on the problem. Without any assumption, the mean would be null. If we are now given a dataset :math:`D = \{(x_1, y_1), (x_2, y_2)\}`, we only consider the function that actually pass through or close to these points, as in the previous figure. This is the learning phase. The more points are added, the more the model will fit the function. Indeed, as we add observations, the error is reduced at these points.

The nature of the covariance matrix is of great importance as it fixes the properties of the functions to consider for inference. This matrix is also called *kernel*. Many covariance functions exist and they can be combined to fit specific needs. A common choice is the squared exponential covariance kernel:

.. math:: k(x, x') = \sqrt{\pi}l \sigma_p^2 \exp{- \frac{(x - x')^2}{2(\sqrt{2}l)^2}},

with :math:`l` the length scale, an hyperparameter, which depends on the magnitudes of the parameters. When dealing with a multidimensional case and non-homogeneous parameters, it is of prime importance to adimentionize everything as one input could bias the optimization of the hyperparameters. 

Then the Gaussian process regression is written as a linear regression

.. math::
    \hat{f}(x_*)&= \sum_{i = 1}^{n}\alpha_i k (x_i, x_*),\\
    \alpha &= (K + \sigma_n^2 I)^{-1}y.

One of the main benefit of this method, is that it provides an information about the variance

.. math::
    \mathbb{V}[f(\mathbf{x}_*)] = k(\mathbf{x}_*, \mathbf{x}_*)-\mathbf{k}(\mathbf{x}_*)^T(K + \sigma_n^2 I)^{-1}\mathbf{k}(\mathbf{x}_*).

The Kriging method is one of the most employed as of today. We can even enhance the result of the regression if we have access to the derivative (or even the hessian) of the function [Forrester2009]_. This could be even more challenging if we don't have an adjoint solver to compute it. Another method is to use a multi-fidelity metamodel in order to obtain an even better solution. This can be performed if we have two codes that compute the same thing or if we have two grids to run from.

.. _pce:

Polynomial chaos expansion
--------------------------

Some citations: [Blatman2009phd]_ [Lemaitreknio2010]_ [Migliorati2013]_ [Sudret2008]_ [Xiu2010]_ [Xiu2002]_

Polynomial chaos expansion (PCE) is a type of surrogate model widely used in uncertainty quantification studies. It takes place in a stochastic framework where model inputs are random variables whose probabilistic distributions determine the families of polynomial regressors. We set out below the details of a PCE construction and its implementation with BATMAN.

Generalities
............

Input scaling
*************

Let :math:`\mathbf{X}=(X_1, X_2, \ldots, X_d)` be the random input vector defined in the input physical space :math:`\mathbb{X}\subset\mathbb{R}^d`. The :math:`i^{\text{th}}` component :math:`X_i` of :math:`\mathbf{X}` is transformed into a new random variable :math:`\zeta_i` obtained by the following centering and scaling operation: 

.. math::

   \tilde{X}_i:=\frac{X_i-\mu_i}{\sigma_i}

where :math:`\mu_i=N^{-1}\sum_{k=1}^Nx_i^{(k)}` and :math:`\sigma_i=\sqrt{(N-1)^{-1}\sum_{k=1}^N\left(x_i^{(k)}-\mu_i\right)^2}` are respectively the empirical mean and standard deviation of :math:`X_i` computed from a :math:`N`-sample :math:`(\mathbf{X}^{(1)},\mathbf{X}^{(2)},\ldots,\mathbf{X}^{(N)})`. The random vector :math:`\tilde{X}=(\tilde{X}_1,\tilde{X}_1,\ldots,\tilde{X}_d)` evolves in a space noted :math:`\tilde{\mathbb{X}}`.

Polynomial expansion
********************

Let :math:`\mathbf{Y}=(Y_1,Y_2,\ldots,Y_p)=f(\mathbf{X})` be the random model output with values in :math:`\mathbb{R}^p`. Assuming that the model output :math:`Y_j` is of finite variance, each component :math:`Y_j` can be considered as a random variable for which there exists a polynomial expansion of the form:

.. math::
   
   Y_j = \displaystyle\sum_{i > 0}\,\gamma_{j,i}\,\Psi_{i}\left(\tilde{\mathbf{X}}\right)=:y_j(\mathbf{X}).

where :math:`\lbrace\Psi_{i}\rbrace_{i\geq 0}` is a basis of orthonormal polynomial functions:

.. math::

   <\Psi_i, \Psi_j> = \delta_{ij}

with :math:`<f, g>\equiv\int_{\tilde{\mathbb{X}}} f(\tilde{\mathbf{x}})g(\tilde{\mathbf{x}}) \rho(\tilde{\mathbf{x}}) \mathrm{d}\tilde{\mathbf{x}}` and :math:`\delta_{ij}` the Kronecker delta function,

and where :math:`\gamma_{j,i}` is the coefficient of the projection of :math:`y_j` onto :math:`\Psi_i`:

.. math::

   \gamma_{j,i}=<y_j, \Psi_i>.

Polynomial basis
****************

In practice, this orthonormal basis is built using the tensor product of :math:`d` 1-D polynomial functions coming from :math:`d` different orthonormal basis: 

.. math::

   \Psi_i(\tilde{\mathbf{x}})=\Psi_{i_1(i)}(\tilde{x}_1)\otimes\Psi_{i_2(i)}(\tilde{x}_2)\otimes\ldots\otimes\Psi_{i_d(i)}(\tilde{x}_d),

where :math:`\left(i_1(i),i_2(i),\ldots,i_d(i)\right)\in\mathbb{N}^d` is the multi-index associated to the integer :math:`i\in\mathbb{N}`. The bijective application :math:`i_{1,2,\ldots,d}=(i_1,i_2,\ldots,i_d):\mathbb{N}\rightarrow\mathbb{N}^d` is an enumerate function to chose (see :ref:`PceEnumerateStategies`).

The choice for the basis functions depends on the probability measure of the random input variables :math:`\zeta_1,\zeta_2,\ldots,\zeta_d`. According to the Askey's scheme, the Hermite polynomials form the optimal basis for random variables following the standard Gaussian distribution:

.. math::

   \forall n\in\mathbb{N},~H_{n+1}(x) = xH_n(x) - nH_{n-1}(x) \text{ with }H_{0}(x)=1\text{ and }H_{1}(x)=x

and the Legendre polynomials are the counterpart for the standard uniform distribution:

.. math::

   \forall n\in\mathbb{N},~L_{n+1}(x) = \frac{2n+1}{n+1}xL_n(x) - \frac{n}{n+1}L_{n-1}(x) \text{ with }L_{0}(x)=1\text{ and }L_{1}(x)=x.

Note that even if standard uniform and Gaussian distributions are widely used to represent input variable uncertainty, the Askey's scheme can also be applied to a wider set of distributions [Xiu2002]_.

Surrogate model
***************

From a deterministic point of view, for a given realization :math:`\mathbf{x}` of :math:`\mathbf{X}` and based on the previous variable change, we have:

.. math::

   y_j\left(\mathbf{x}\right):=\displaystyle\sum_{i \geq 0}\,\gamma_{j,i}\,\Psi_{i}\left(\tilde{\mathbf{x}}\right).

In practice, we use a truncation strategy (see :ref:`PceTruncationStategies`) limiting this polynomial expansion to the more significant elements in terms of explained output variance:

.. math::

   \hat{y}_j\left(\mathbf{x}\right):=\displaystyle\sum_{i = 0}^r\,\gamma_{j,i}\,\Psi_{i}\left(\tilde{\mathbf{x}}\right).

Thus, :math:`\hat{\mathbf{y}}=(\hat{y}_1,\hat{y}_2,\ldots,\hat{y}_p)` is a surrogate model of :math:`\mathbf{y}=(y_1,y_2,\ldots,y_p)`.

Properties
..........

Various statistical moments associated to the PC surrogate model have explicit formulations, thus avoiding Monte-Carlo sampling, even if this metamodel is computationally cheap.

For the :math:`j^{\text{th}}` output, the expectation reads:

.. math::

   \mathbb{E}\left[\hat{y}_j\left(\mathbf{X}\right)\right]=\gamma_{j,0}

For the :math:`j^{\text{th}}` output, the variance reads:

.. math::

   \mathbb{V}\left[\hat{y}_j\left(\mathbf{X}\right)\right]=\sum_{i = 1}^r\gamma_{j,i}^2

For the :math:`j^{\text{th}}` and :math:`k^{\text{th}}` outputs, the expectation reads:

.. math::

   \mathbb{C}\left[\hat{y}_j,\hat{y}_k\left(\mathbf{X}\right)\right]=\sum_{i = 1}^r\gamma_{j,i}\gamma_{k,i}

In the context of global sensitivity analysis, there are similar results for the Sobol' indices [Sudret2008]_.

Options
.......

.. _PceEnumerateStategies:

Enumerate strategies
********************

Remind that:

- :math:`\forall i\in\{0,1,\ldots,r\},~\Psi_i(\tilde{\mathbf{x}})=\Psi_{1,i_1(i)}(\tilde{x}_1)\otimes\Psi_{2,i_2(i)}(\tilde{x}_2)\otimes\Psi_{d,i_d(i)}(\tilde{x}_d)`.
- :math:`\forall k\in\{1,2,\ldots,d\},~\{\Psi_{k,i}\}_{0\leq i \leq P}` are the :math:`P+1` first elements of the polynomial basis associated to :math:`X_k` and their degrees are lower or equal to :math:`P`.
- :math:`\forall i\in\{0,1,\ldots,r\},~\forall k\in\{1,2,\ldots,d\},~\text{degree}(\Psi_{k,i_k(i)})=i_k(i)\leq P`.
- :math:`\forall i\in\{0,1,\ldots,r\},~\text{degree}(\Psi_{i})=\sum_{k=1}^d\text{degree}(\Psi_{k,i_k(i)})\leq P`.

An enumerate function is a bijective application from :math:`\{0,1,\ldots,P\}` to :math:`\{0,1,\ldots,P\}^d` of the form:

.. math::

   i\mapsto i_{1,2,\ldots,d}(i)=(i_1(i),i_2(i),\ldots,i_d(i)).

The bijectivity property implies that the initial term is:

.. math::

   i_{1,2,\ldots,d}(0)=\{0,0,\ldots,0\}

and the next ones satisfy the constraint:

.. math::

   \forall 0\leq i \leq j,~ \text{degree}(\Psi_i)<\text{degree}(\Psi_j) \Leftrightarrow \forall 0\leq i \leq j,~ \sum_{k=1}^d i_k(i) < \sum_{k=1}^d i_k(j)

or 

.. math::

   \forall 1\leq i \leq j,~\exists m \in\{1,2,\ldots,d\}:~(\forall k\leq m,~i_k(i)=i_k(j))~\text{ and }~ (i_m(i)<i_m(j)).

**Linear enumerate function**

A natural linear enumerate strategy is the lexicographical order with a constraint of increasing total degree. The unique requirement of this strategy is the input space dimension :math:`d`.


**Hyperbolic anisotropic enumerate function**

Hyperbolic truncation strategy gives an advantage to the main effects and low-order interactions. From a multi-index point of view, this selection implies to discard multi-indices including an important number of non-zeros components.

:math:`\forall q \in ]0, 1]`, the anisotropic hyperbolic norm of a multi-index :math:`\boldsymbol{\alpha}\in\mathbb{R}^d` is defined by:

.. math::

   \| \boldsymbol{\alpha} \|_{\mathbf{w}, q} = \left( \sum_{k=1}^{d} w_k \alpha_k^q \right)^{1/q}

where the :math:`w_k`‘s are real positive numbers. In the case where :math:`\mathbf{w}=(1,1,\ldots,1)`, the strategy is isotropic.

.. _PceTruncationStategies:

Truncation strategies
*********************

In this section, we present different truncation strategies.


Note that for small :math:`d`, advanced truncation strategies that consist in eliminating high-order interaction terms or using sparse structure [Blatman2009phd]_ [Migliorati2013]_ are not necessary.

**Fixed truncation strategy**

The standard truncation strategy consists in constraining the number of terms :math:`r` by the number of random variables :math:`d` and by the total polynomial degree :math:`P` of the PCE. Precisely, the choice of :math:`r` is equal to:

.. math::

   r = \frac{(d + P)!}{d!\,P!}.

All the polynomials :math:`\Psi_i` involving the :math:`d` random variables with a total degree less or equal to :math:`P` are retained in the PC expansion. Then, the PC approximation of :math:`y_j` is formulated as:

.. math::

   \widehat{y}_j(\mathbf{x}) = \displaystyle\sum_{0\leq i \leq r}\,\gamma_{j,i}\,\Psi_i\left(\tilde{\mathbf{x}}\right) = \displaystyle\sum_{i\in\mathbb{N}\atop\sum_{k=1}^di_k(i)\leq P}\,\gamma_{j,i}\,\Psi_i\left(\tilde{\mathbf{x}}\right).

.. warning:: The number of terms :math:`r` grows polynomially both in :math:`d` and :math:`P` though, which may lead to difficulties in terms of computational efficiency and memory requirements when dealing with high-dimensional problems.

**Sequential truncation strategy**

The sequential strategy consists in constructing the basis of the truncated PC iteratively. Precisely, one begins with the first term :math:`\Psi_0`, that is :math:`K_0 = \{0\}`, and one complements the current basis as follows: :math:`K_{i+1} = K_i \cup \{\Psi_{i+1}\}`. The construction process is stopped when a given accuracy criterion is reached, or when :math:`i` is equal to a prescribed maximum basis size :math:`r`.

**Cleaning truncation strategy**

The cleaning strategy aims at building a PC expansion containing at most :math:`r` significant coefficients, i.e. at most :math:`r` significant basis functions. It proceeds as follows:

- Generate an initial PC basis made of the :math:`r` first polynomials (according to the adopted EnumerateFunction), or equivalently an initial set of indices :math:`\mathcal{I} = \{0, \ldots, r-1\}`.
- Discard from the basis all those polynomials :math:`\Psi_i` associated with insignificance coefficients, i.e. the coefficients that satisfy:

.. math::

   |\gamma_i| \leq \epsilon \times \max_{ i' \in \mathcal{I} } |\gamma_{i'}|

where :math:`\epsilon` is the significance factor, default is :math:`\epsilon = 10^{-4}`.

- Add the next basis term :math:`\Psi_{i+1}` to the current basis :math:`\mathcal{I}`.
- Reiterate the procedure until either :math:`r` terms have been retained or if the given maximum index :math:`r_{\text{max}}` has been reached.

Coefficient calculation strategies
**********************************

We focus here on non-intrusive approaches to numerically compute the coefficients :math:`\lbrace\gamma_{j,i}\rbrace_{0\leq i \leq r\\ 1\leq j \leq p}` using :math:`N` snapshots from :math:`\mathcal{D}_N`. 

**Least-square strategy**

Based on a :math:`N`-sample :math:`\left(\mathbf{X}^{(k)},\mathbf{Y}^{(k)}\right)_{1\leq k \leq N}`, the least-square strategy seeks the solution of the optimization problem:

.. math::

   \hat{\boldsymbol{\gamma}}=\underset{\boldsymbol{\gamma}\in\mathbb{R}^r}{\operatorname{argmin}} \sum_{k=1}^N \sum_{j=1}^p \left(y_j^{(k)}-\displaystyle\sum_{0\leq i \leq r}\,\gamma_{j,i}\,\Psi_i\left(\tilde{\mathbf{x}}^{(k)}\right)\right)^2

which is achieved through classical linear algebra tools.

Note that the sample size :math:`N` required by this strategy is at least equal to :math:`p(r+1)` where :math:`r` is the number of PC coefficients and :math:`p` is the output vector dimension.

**Quadrature strategy**

The spectral projection relies on the orthonormality property of the polynomial basis. For the :math:`j^{\text{th}}`, the :math:`i^{\text{th}}` coefficient :math:`\gamma_{j,i}` is computed using a Gaussian quadrature rule as:

.. math::

   \gamma_{j,i} = <y_j,\Psi_i> \,\cong\,\displaystyle\sum_{k = 1}^{N}\,y_j^{(k)}\,\Psi_i(\tilde{\mathbf{x}}^{(k)})\,w^{(k)}

where:

- :math:`\mathbf{y}^{(k)} = \mathcal{M}(\mathbf{x}^{(k)})` is the evaluation of the simulator at the :math:`k^{\text{th}}` quadrature root :math:`\tilde{\mathbf{x}}^{(k)}` of :math:`\Psi_i`,
- :math:`w^{k}` is the weight associated to :math:`\mathbf{x}^{(k)}`. 

Note that the number of quadrature roots required in each uncertain direction to ensure an accurate calculation of the integral :math:`<y_j,\Psi_i>` is equal to :math:`(P+1)`.

Implementation
..............

Concerning the library BATMAN, polynomial chaos expansion has to be specified in the block **surrogate** of the JSON settings file, setting the keyword **method** at **pc** and eventually using ones of the following keywords:

+-------------------+--------------------------------------+----------------------------------------------------------------------------------------------------------+
| Keywords          | Type                                 | Description                                                                                              |
+===================+======================================+==========================================================================================================+
| **strategy**      | string                               | Strategy for the weight computation:                                                                     |
|                   |                                      |                                                                                                          |
|                   |                                      | - least square: **LS**                                                                                   |
|                   |                                      | - quadrature: **Quad**                                                                                   |
+-------------------+--------------------------------------+----------------------------------------------------------------------------------------------------------+
| **degree**        | integer                              | Total polynomial degree :math:`P`                                                                        |
+-------------------+--------------------------------------+----------------------------------------------------------------------------------------------------------+
| **distributions** | list(:class:`openturns.Distribution`)| Distributions of each input parameter                                                                    |
+-------------------+--------------------------------------+----------------------------------------------------------------------------------------------------------+
| **n_sample**      | integer                              | Number of samples for least square                                                                       |
+-------------------+--------------------------------------+----------------------------------------------------------------------------------------------------------+

Multifidelity
-------------

It is possible to combine several level of fidelity in order to lower the computational cost of the surrogate
building process. The fidelity can be either expressed as a mesh difference, a convergence difference, or even a
different set of solvers. [Forrester2006]_ proposed a way of combining these fidelities by building a low
fidelity model and correct it using a model of the error:

.. math:: \hat{f}(x) = f_c(x) + \hat{f}_{\epsilon}(f_e(x), f_c(x)),

with :math:`\hat{f}_{\epsilon}` the surrogate model representing the error between the two fidelity levels.
This method needs nested design of experiments for the error model to be computed.

Considering two levels of fidelity :math:`f_e` and :math:`f_c`, respectively an expensive and a cheap function expressed as a computational cost. A cost ratio :math:`\alpha` between the two can be defined as:

.. math:: \alpha = \frac{f_e}{f_c}.

Using this cost relationship an setting a computational budget :math:`C`, it is possible to get a relation between the number of cheap and expensive realizations:

.. math:: C f_e &= N_e f_e + N_c f_c,\\
          C f_e &= N_e f_e + N_c\frac{\alpha}{f_e},\\
          C &= N_e + N_c\alpha, \\
          N_c &= \frac{C - N_e}{\alpha}.

As the design being nested, the number of cheap experiments must be strictly superior to the number or expensive ones. Indeed, the opposite would result in no additional information to the system.

Mixture of expert
-----------------

The prediction based on surrogate models can be false when a bifurcation occurs because of the inherent way it is built. A solution proposed to this problem is called the Local Decomposition Method. This method is used to achieve a proper separation of the behaviours of the bifurcation by dividing the input parameter space in sub-spaces called clusters using unsupervised machine learning. Then, local surrogate models are computed on these sub-spaces. However, in order to predict data for new sample points, these need to be classified into their respective cluster using supervised machine learning. Their data are then predictable according to their cluster and local surrogate model.

This method is done in several steps: 

1. Computation of PCA (see documentation on POD) on data of higher dimensions than scalars. This computation will compress information to a small number of components so the clusterer will have less problems trying to identify patterns in the data than with a great number of components and allow better results.
2. Clustering using unsupervised machine learning tools from Scikit-Learn.
3. Formation of local surrogate models.
4. Classification using supervised machine learning tools from Scikit-Learn.
5. Prediction of new points using the local models according to their cluster affiliations.

Clustering using Unsupervised Machine Learning
..............................................

Unsupervised machine learning tools from Scikit-Learn are used to cluster data set according to parameters chosen by the user, especially the number of clusters (documentation: `ScikitLearn Documentation  <http://scikit-learn.org/stable/modules/clustering.html#clustering>`_). The objective of these methods is to put a label on each data according to their affiliation to a cluster. To cover the majority of possible data types, different default methods are used.
The method of *KMeans* proposes a fast computation time and simple algorithm while *GaussianMixture* use probabilities instead of metrics and provides good results and *DBSCAN* is based on a density-based algorithm:

**KMeans**

*KMeans* clustering algorithm use iterative refinement to form clusters. As an input it needs at least the number of clusters (documentation: `KMeans API <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans>`_).
The center of clusters called centroids is a parameter that is either randomly generated initially, given by the user or generated by a special scheme to allow for better results. The algorithm then iterates between two steps: 

1. Data assignment step: 

The clusters are defined by their centroid. With this step, each data point is assigned to its nearest centroid by taking the minimum of the squared Euclidean distance.

2. Centroid update step:

The centroid are then recomputed. To do so, the mean of all distances between data points assigned to that centroid's cluster is taken:

The algorithm iterates between steps one and two until it meets the stopping criteria (maximum number of iteration or minimization of distances) where it converge to a result. To avoid falling in a local optimum, it is possible to do several centroid initialisations, the algorithm will then chose the best result of these.

The method of *MiniBatchKMeans* can also be used (documentation: `MiniBatchKMeans API <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans>`_). This method is a variant of *KMeans* which uses mini batch by picking randomly some samples from the dataset during the first step of the algorithm. It allows faster convergence time but reduce the quality of results.

**DBSCAN**

*DBSCAN* clustering algorithm is a density-based algorithm for clustering purposes (documentation: `DBSCAN API <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN>`_). As an input it needs at least the density :math:`\epsilon` and the integer *MinPts*.

It uses the concept of :math:`\epsilon`-neighbourhood which define the density by considering the data points within a radius :math:`\epsilon` for each data point. Each point can be low-density or high-density depending on the parameter *MinPts* which represents a minimum number of neighbours and fix the criteria of this classification by comparing it to the :math:`\epsilon`-neighbourhood of each point.

The two value of :math:`\epsilon` and *MinPts* can then further classify each point:
- Core points which have at least *MinPts* points in their neighbourhood. These tend to be in the interior of clusters and are part of the fondation of a cluster.
- Border points who have less than *MinPts* points in their neighbourhood but are in the :math:`\epsilon`-neighbourhood of a core point.
- Noise point who are neither of the two precedents.

There are then 3 cases of reachability:
- A point directly density-reachable refers to a point recheable in the :math:`\epsilon`-
neighbourhood of a core point.
- A point density-reachable exists when there is a chain of points that allow you to direcly reach this point.
- A point is density-connected to another point if they are density-reachable from a center.

The algorithm is then as follows: 

1) Computation of Euclidean distances between each point and determine if a point is in the :math:`\epsilon`-neighbourhood of an other.
2) Assignment to a cluster if the point validate the *MinPts* criteria and expand the cluster by checking if the :math:`\epsilon`-neighbour points also validate the *MinPts* criteria.

**Gaussian Mixture**

*Gaussian Mixture* algorithm uses probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters (documentation: `GMM API <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_).

Each cluster :math:`C_k` coming from *q* clusters has a probability distribution of parameter :math:`\theta_k` and proportion :math:`w_k`(corresponding to the mean :math:`\mu_k` an variance :math:`\sigma^2_k`). These probability distribution are regrouped in a mixture parameter :math:`\Phi`.
*Gaussian Mixture Model* (GMM) consists in modeling the dataset with a mixture distribution of multivariate normal distributions *g*. Each one is associated to a cluster such that probability density function of a point :math:`B_i` (with *n* the number of samples) is:

.. math::
	p(B_i\mid \Phi)=\sum_{k=1}^{q} w_k g(B_i \mid  \theta_k ), \forall i \in [1,n]

the mixture weights :math:`w_i` represents the probability that the observation comes from the *k-th* Gaussian Distribution and :math:`\theta_k` gives the mean and the covariance of the multivariate normal distribution *g*. 

Then an Expectation Maximization algorithm (EM) is used on :math:`\Phi`. Bayes theorem can express the expectation of the posterior probability :math:`\gamma_k` of belonging to the cluster *k* (E-step): 

.. math::
	\gamma_k(B_i)=p(C_k\mid B_i)=\frac{p(B_i \mid C_k)p(C_k)}{p(B_i)}=\frac{w_k \, g(B_i\mid\theta_k)}{\sum_{l=1}^{q}w_l\, g(B_i \mid \theta_l)}, \forall k \in [1,q], \forall i \in [1,n].

Then, the mixture parameters can be re-estimated (M-step): 

.. math::
	\mu_k =\frac{\sum_{i=1}^{N}\gamma_k(B_i).B_i}{\sum_{i=1}^{N}\gamma_k(B_i)}, \forall k \in [1,q],

.. math::
	\sum_k=\frac{\sum_{i=1}^{N}\gamma_k(B_i).(B_i-\mu_k).(B_i-\mu_k)^T}{\sum_{i=1}^{N}\gamma_k(B_i)}, \forall k \in [1,q],

.. math::
	w_k = \frac{1}{n} \sum_{i=1}^{N}\gamma_k(B_i), \forall k \in [1,q].

The algorithm iterates between these two steps until the convergence of the log likelihood. The cluster of each data point can be determined using the previous probability expression and taking the maximum result as the most probable cluster of affiliation.

Predictions using Supervised Machine Learning
.............................................

The labels of clusters and their data points can be used to form local surrogate models for each of these clusters (documentation: `ScikitLearn Documentation <http://scikit-learn.org/stable/supervised_learning.html>`_).
These method used for local surrogate modelling are the same used for global models as detailled before.

In order to make data predictions for new samples, supervised machine learning tools from Scikit-Learn are used to affiliate new points to their clusters then use the local models to predict the data. These classifiers use the previously clusterised data set along with the labels (from unsupervised machine learning) to classify new data points called training set by applying what they learnt with the previously clusterised data set called trained set. To cover the majority of possible data types, different default methods are used.
The methods of *GaussianNaiveBayes*, *SupportVectorMachine*, *K-NearestNeighbours* and *GaussianProcess* are proposed here: 

**Gaussian Naive Bayes**

*Gaussian Naive Bayes* (GNB) is a method that use Bayes' theorem with the so called "naive" assumption of indepedance between each pair of features (documentation: `GaussianNB API <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB>`_).
This can be translated mathematically with a class variable *y* (cluster) and a dependent feature vector *(x1,...,xn)* as:

.. math::
	P(x_i\mid y,x_1,...,x_n)=P(x_i \mid y),

which then with the Bayes theorem: 

.. math::
	P(y \mid x_1,...,x_n)=\frac{P(y)P(x_1,...,x_n \mid y)}{P(x_1,...,x_n)} \Rightarrow P(y \mid x_1,...,x_n)=\frac{P(y)\prod_{i=1}^{n}P(x_i\mid y)}{P(x_1,...,x_n)}.

The probability :math:`P(x_1,...x_n)` is constant given the input which implicates that :math:`P(y \mid x_1,...,x_n)` is proportionnal to the numerator so:

.. math::
	\hat{y}=arg \underset{y}{max} P(y) \prod_{i=1}^{n}P(x_i \mid y).

The Maximum A Posteriori (MAP) estimator is applicable to estimate P(y) and P(:math:`x_i \mid y`)
For *GNB* the likelihood of the features is assumed to be Gaussian:

.. math::
	P(x_i \mid y)=\frac{1}{\sqrt{2 \pi \sigma^2_y}}exp(-\frac{(x_i \, - \, \mu_y)^2}{2 \sigma^2_y}).

The algorithm then works as follow: 

1. Computation of likelihood table by finding the probabilities,
2. Naive Bayesian equations to calculate the posterior probability for each class, the class with the highest probability is the outcome of prediction.

**Support Vector Machine**

*Support Vector Classification* (SVC) is a method that constructs a hyper-plane or a set of hyper-planes in a high or infinite dimensional space to do classification (documentation: `SVC API <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_). A good separation is achieved by the hyper-plane that has the largest distance to the nearest trained data points of any class. Different Kernel functions can be used for the decision function.

Important parameters such as the margin to points or the regularization parameter which allow more or less the misclassification of a training point can vary the results significantly.

**K Nearest-Neighbours**

*K Nearest-Neighbours* (KNN) is a classifier that works on the nearest neighbours to a point by computing its distance to every other points (documentation: `KNN API <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_). The parameter K is then used to train the training points.

The algorithm is as follow for a training point: 

1. Computing of the Euclidean distance to trained point,
2. Sort the distance values,
3. Use the parameter K to get the top K rows of datas,
4. Get the most frequent class of these rows to predict the class of the training point.

These steps are repeated for each training point in order to classify the training set.

**Gaussian Process**

*Gaussian Process Classification* (GPC) is a classifier based on class probabilities to make predictions (documentation: `GPC API <http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier>`_). 

RAE Example of Mixture
----------------------

Step 1: CFD Runs
................

For this example, we use the code *elsA* co-developed at *ONERA* and CERFACS to run CFD simulations on a wing profile. The code *elsA* is a CFD simulation platform that deals with internal and external aerodynamics from the low subsonic to the high supersonic flow regimes. It relies on solving the compressible 3-D Navier-Stokes equations and can allows the simulation of the flow around complex geometries (see documentation here: `ELSA <http://elsa.onera.fr/>`_).
The geometry used for this case is the *RAE2822* wing profile from the AS28G aircraft configuration that you can see here: 

.. image:: ../fig/mesh_rae.png

.. image:: ../fig/shock_rae.pdf

This test case is a basic wing profile used to study the interactions between boundary layers and shocks. These interactions can lead to the detachment of the boundary layer which imply lift problems for the wing. To improve the detachment, we want the shock to be as far as possible on the wing to limit its influence on the pressure around the profile.

Here, we use the code *elsA* to run simulations based on this geometry to simulate a complex steady flow around the profile. These simulations are based on different inflow conditions characterized with the incidence angle of the inflow and its Mach number to form a set of snapshots and a Design of Experiment (DoE). These CFD runs are expensive so we want to construct a surrogate model to predict new set of data for future applications. However, the inflow conditions can lead to the presence of a shock or not on the profile. Thus, to resolve the problem of bifurcation, we use the LDM previously described

Step 2: Application of Local Decomposition Method
.................................................

To properly seperate our data in clusters, we need to use a physical-based sensor that can detects shocks along the wing according to the pressure calculated by *elsA*. Here, we use a physical-based sensor called the *Jameson Shock sensor* that can detects discontinuities and nonlinearities:

.. math:: \mu_i = \frac{\abs{P_{i+1}-2P_i + P_{i-1}}}{\eps_0 + \abs{P_{i+1}} + 2\abs{P_i} + \abs{P_{i-1}}},

with :math:`P_i` the pressure along the wing using the curvilinear abscissa (in this case the Pressure but it can be the entropy) and the constant :math:`\eps_0` to avoid being close to zero of the denominator. This sensor is a central element of the method to achieve a proper separation of the physical regimes.
It is computed on all snapshots. Its dimension is reduced by the use of *Principal Component Analysis* (describe in POD documentation) to ease the clustering step and provide a set of data characterizing the physical regimes for each snapshot.

Then, we use a unsupervised machine-learning tool such as *KMeans* or *Gaussian Mixture* to separate the different regimes. The set of data provided by *elsA* serves as an input for the algorithm and each snapshot is associated to its cluster based on the features of the sensor. We can then plot the boundaries over the DoE to visualize the results given by the clustering step:

.. image:: ../fig/doe_rae.png

These clusters are used to form local surrogate models using for example *Kriging* or *Polynomial Chaos* methods (previously described). The quality of these models is assessed with a LOOCV method. A resampling for the worst cluster could be possible to improve the accuracy of the model. These local models allow for a more accurate prediction than a global model who would have tried to model the bifurcation. Indeed, local models give the lowest q2 value at *0.8357* while a global model gives us a q2 value at *0.5974*

In order to form predictions for new DoE points, we use a supervised machine-learning tool like *Support Vector Machine* or *Gaussian Process Classification* to classify these new points by fitting the algorithm with the previously set of DOE points and their associated cluster. These algorithms then return back the classification of the new points to the previous clusters. A prediction of the new points is now possible by using the local models previously formed according to the right cluster. These figure show the prediction of a shock and non shock pressure along the wing profile compared to simulations:

.. image:: ../fig/doe_rae.png
