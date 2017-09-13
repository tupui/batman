"""
Probability Density Function tools
----------------------------------
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.optimize import differential_evolution


def kernel_smoothing(data, optimize=False):
    """Create gaussian kernel.

    :param bool optimize: use global optimization of grid search
    :return: gaussian kernel
    :rtype: :class:`sklearn.neighbors.KernelDensity`
    """
    n_samples, dim = data.shape
    cv = n_samples if n_samples < 50 else 50

    if optimize:
        def bw_score(bw):
            score = cross_val_score(KernelDensity(bandwidth=bw),
                                    data, cv=cv, n_jobs=-1)
            return - score.mean()

        bounds = [(0.1, 5.)]
        results = differential_evolution(bw_score, bounds, maxiter=5)
        bw = results.x
        ks_gaussian = KernelDensity(bandwidth=bw)
        ks_gaussian.fit(data)
    else:
        scott = n_samples ** (-1. / (dim + 4))
        silverman = (n_samples * (dim + 2) / 4.) ** (-1. / (dim + 4))
        bandwidth = np.hstack([np.linspace(0.1, 5.0, 30), scott, silverman])
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': bandwidth},
                            cv=cv, n_jobs=-1)  # 20-fold cross-validation
        grid.fit(data)
        ks_gaussian = grid.best_estimator_

    return ks_gaussian
