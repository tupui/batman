.. _pod:
.. currentmodule:: batman.pod

POD for *Proper Orthogonal Decomposition*
=========================================

What is it?
-----------

The *Proper Orthogonal Decomposition* (POD) is a technique used to decompose a matrix and characterize it by its principal components which are called modes [AnindyaChatterjee2000]_. To approximate a function :math:`z(x,t)`, only a finite sum of terms is required:

.. math::
   z(x,t) \simeq \sum_{k=1}^{m} a_k(t) \phi_k(x).

The function :math:`\phi_{k}(x)` have an infinite representation. It can be chosen as a Fourier series or Chebyshev polynomials, etc. For a chosen basis of function, a set of unique time-functions :math:`a_k(t)` arise. In case of the POD, the basis function are orthonormal. Meaning that:

.. math::
   \int_{x} \phi_{k_1} \phi_{k_2} dx &= \left\{\begin{array}{rcl} 1 & \text{if} & k_1 = k_2   \\ 0 & \text{if} & k_1 \neq k_2\end{array}\right. ,\\
   a_k (t) &= \int_{x} z(x,t) \phi_k(x) dx.

The principle of the POD is to choose :math:`\phi_k(x)` such that the approximation of :math:`z(x,t)` is the best in a least squares sense. These orthonormal functions are called the *proper orthogonal modes* of the function.

When dealing with CFD simulations, the size of the domain :math:`m` is usually smaller than the number of measures, snapshots, :math:`n`. Hence, from the existing decomposition methods, the *Singular Value Decomposition* (SVD) is used. It is the snapshots methods [Cordier2006]_.

The Singular Value Decomposition (SVD) is a factorization operation of a matrix expressed as:

.. math::
   A = U \Sigma V^T,

with :math:`V` diagonalizes :math:`A^TA`, :math:`U` diagonalizes :math:`AA^T` and :math:`\Sigma` is the singular value matrix which diagonal is composed by the singular values of :math:`A`. Knowing that a singular value is the square root of an eigen value. :math:`u_i` and :math:`v_i` are eigen vectors of respectively :math:`U` and :math:`V` which form an orthonormal basis. Thus, the initial matrix can be rewritten:

.. math::
   A = \sum_{i=1}^{r} \sigma_i u_i v_i^T,

:math:`r` being the rank of the matrix. If taken :math:`k < r`, an approximation of the initial matrix can be constructed. This allows to compress the data as only an extract of :math:`u` and :math:`v` need to be stored.
