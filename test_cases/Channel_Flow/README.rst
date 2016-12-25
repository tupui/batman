Channel Flow
============

This test case uses the Channel Flow function.

.. math::
    dh/ds &= I (1-(h/h_n)^-10/3)/(1 - (h/h_c)^-3)\\
    h_c &= (q^2/g)^1/3\\
    h_n &= (q^2/IK_s^2)^3/10

The dimension of the output is 400.

This tast case shows how with `-u` option we can propagate uncertainties and retrieve a Sobol' map
and aggregated indices.

The parameters of the analysis are to be configured within
`settings.json`.
