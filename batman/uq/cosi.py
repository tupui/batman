"""
Cosine transformation indices
-----------------------------

Plischke E., How to compute variance-based sensitivity indicators with your
spreadsheet software, Environmental Modelling & Software,
2012. DOI: 10.1016/j.envsoft.2012.03.004
"""
import numpy as np
from scipy.fftpack import dct


def cosi(sample, data):
    """Cosine transformation sensitivity.

    Use Discret Cosine Transformation (DCT) to compute sensitivity indices.

    :param array_like sample: Sample of parameters of Shape
      (n_samples, n_params).
    :param array_like data: Sample of realization which corresponds to the
      sample of parameters :attr:`sample` (n_samples, ).
    :returns: First order sensitivity indices.
    :rtype: (Sobol, n_features).
    """
    sample = np.asarray(sample)
    data = np.asarray(data).flatten()

    ns, dim = sample.shape

    n_coeffs = int(max(np.ceil(np.sqrt(ns)), 3))

    s_indices = []
    for d in range(dim):
        idx = sample[:, d].argsort()
        data_r = data[idx]

        coeffs = dct(data_r)  # cosine transformation frequencies

        # Do not take the first coefficient which is the mean
        var_u = sum(coeffs[1:] ** 2)
        var_c = sum(coeffs[1:n_coeffs] ** 2)
        s_c = var_c / var_u

        s_indices.append(s_c)

    return s_indices
