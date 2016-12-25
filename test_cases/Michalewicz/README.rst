Michalewicz
===========

This test case uses the Michalewicz function.

It is a multimodal *d*-dimensional function which has :math:`d!` local minima - for this *test-case*: 

.. math:: f(x)=-\sum_{i=1}^d \sin(x_i)\sin^{2m}\left(\frac{ix_i^2}{\pi}\right),

where *m* defines the steepness of the valleys and ridges.

.. note:: + It is to difficult to search a global minimum when :math:`m` reaches large value. Therefore, it is recommended to have :math:`m < 10`.
          + In this case we used the two-dimensional form, i.e. :math:`d = 2`. 

To summarize, we have the Michalewicz 2*D* function as follows:

.. math:: f(x)=-\sin(x_1)\sin^{20}\left(\frac{x_1^2}{\pi}\right)-\sin(x_2)\sin^{20}\left(\frac{2x_2^2}{\pi}\right).
