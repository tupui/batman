"""
Generic dataset
---------------

"""
import logging
import numpy as np
from scipy.spatial import distance
from .utils import multi_eval
from ..space import Sample


class DbGeneric(object):
    """Generic database class."""

    logger = logging.getLogger(__name__)

    def __init__(self, space=None, data=None, fnames=None):
        """Handle generic database function.

        From a given set of input parameters, it gets the closest point from
        the database.

        :param array_like space: Input parameters,
          shape (n_samples, n_features).
        :param array_like data: Output corresponding to the input parameters,
          shape (n_samples, n_features).
        :param lst(str) fnames: [input, output] files names.
        """
        if fnames is not None:
            self.space = np.load(fnames[0])
            self.data = np.load(fnames[1])
        else:
            self.space = np.asarray(space)
            self.data = np.asarray(data)

        self.d_out = self.data.shape[1]
        self.ns, self.d_in = self.space.shape

    def __repr__(self):
        return ("Generic dataset: N_s = {}, d_in = {} -> d_out = {}"
                .format(self.ns, self.d_in, self.d_out))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param array_like x: inputs (1, n_features).
        :return: f(x).
        :rtype: array_like (1, n_features).
        """
        dists = distance.cdist([x, x], self.space, 'seuclidean')
        idx = np.argmin(dists, axis=1)
        idx = idx[0]

        corresp = self.space[idx]
        self.logger.debug("Input: {} -> Database: {}".format(x, corresp))

        return self.data[idx]
