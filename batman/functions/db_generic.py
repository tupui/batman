"""
Generic dataset
---------------

"""
import logging
import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
from .utils import multi_eval


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

        # Accomodate for discrepancy in parameters' ranges
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.space)
        self.sample_scaled = self.scaler.transform(self.space)

    def __repr__(self):
        return ("Generic dataset: N_s = {}, d_in = {} -> d_out = {}"
                .format(self.ns, self.d_in, self.d_out))

    @multi_eval
    def __call__(self, x, full=False):
        """Call function.

        :param array_like x: inputs (1, n_features).
        :param bool full: Whether to return the sample from the database.
        :return: f(x).
        :rtype: array_like (1, n_features).
        """
        x_scaled = self.scaler.transform(np.array(x)[None, :])
        dists = distance.cdist(x_scaled, self.sample_scaled, 'seuclidean')
        idx = np.argmin(dists, axis=1)

        corresp = self.space[idx]
        self.logger.debug("Input: {} -> Database: {}".format(x, corresp))

        return (corresp, self.data[idx]) if full else self.data[idx]
