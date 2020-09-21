# coding: utf8
"""
Driver Class
============

Defines all methods used to interact with other classes.

:Example:

::

    >> from batman import Driver
    >> driver = Driver(settings, output_path)
    >> driver.sampling_pod(update=False)
    >> driver.write_pod()
    >> driver.prediction(write=True)
    >> driver.write_model()
    >> driver.uq()
    >> driver.visualization()

"""
import logging
import os
import pickle
from copy import copy
from concurrent import futures
import numpy as np
import sklearn.gaussian_process.kernels as kernels

from .misc.schema import Settings, Sampling
from .pod import Pod
from .space import (Space, Sample, dists_to_ot)
from .surrogate import SurrogateModel
from .tasks import (ProviderFunction, ProviderFile, ProviderJob)
from .uq import UQ
from .visualization import (response_surface, Kiviat3D, mesh_2D)
from .functions.utils import multi_eval


class Driver:
    """Driver class."""

    logger = logging.getLogger(__name__)
    # Structure of the output directory
    fname_tree = {
        'snapshots': 'snapshots',
        'space': 'space',
        'data': 'data.dat',
        'pod': 'surrogate/pod',
        'surrogate': 'surrogate',
        'predictions': 'predictions',
        'uq': 'uq',
        'visualization': 'visualization',
    }

    def __init__(self, settings: Settings, fname: str):
        """Initialize Driver.

        From settings, init snapshot, space [POD, surrogate].

        :param settings: settings.
        :param fname: output folder path.

        """
        self.settings = settings
        self.fname = fname
        os.makedirs(self.fname, exist_ok=True)

        # Space preprocessing
        try:
            init_size = self.settings.space.sampling.init_size
        except AttributeError:  # when providing DoE as a list
            init_size = self.settings.space.sampling

        duplicate = True if self.settings.space.sampling == 'saltelli' else False

        if self.settings.surrogate.multifidelity:
            multifidelity = [self.settings.surrogate.cost_ratio,
                             self.settings.surrogate.grand_cost]
        else:
            multifidelity = None

        self.space = Space(self.settings.space.corners,
                           init_size,
                           nrefine=self.settings.space.resampling.resamp_size,
                           plabels=self.settings.snapshot.plabels,
                           psizes=self.settings.snapshot.psizes,
                           multifidelity=multifidelity,
                           duplicate=duplicate,
                           gp_samplers=None)

        # Data Providers
        setting_provider = self.settings.snapshot.provider.dict()
        provider_type = setting_provider.pop('type')
        args = {'discover_pattern': setting_provider.pop('discover', None)}
        self.logger.info('Select data provider type "{}"'.format(provider_type))
        if provider_type == 'function':
            provider = ProviderFunction
        elif provider_type == 'file':
            provider = ProviderFile
        else:  # provider_type == 'job':
            provider = ProviderJob
            args['save_dir'] = os.path.join(self.fname, self.fname_tree['snapshots'])
            args['pool'] = futures.ThreadPoolExecutor(
                max_workers=self.settings.snapshot.max_workers)

        args.update(setting_provider)
        args.update(self.settings.snapshot.io)

        self.provider = provider(plabels=self.settings.snapshot.plabels,
                                 flabels=self.settings.snapshot.flabels,
                                 psizes=self.settings.snapshot.psizes,
                                 fsizes=self.settings.snapshot.fsizes,
                                 **args)
        self.snapshot_counter = 0  # TODO make it useless

        # Fill space
        if isinstance(self.settings.space.sampling, list):
            # a list of points is provided
            self.logger.info('Reading list of points from the settings.')
            self.space += self.settings.space.sampling
        elif provider_type == 'file':
            self.space += self.provider._cache.space
        elif isinstance(self.settings.space.sampling, Sampling):
            # use sampling method
            self.space.sampling(self.settings.space.sampling.init_size,
                                self.settings.space.sampling.method,
                                self.settings.space.sampling.distributions,
                                self.settings.space.sampling.discrete)

        self.to_compute_points = self.space.values

        # Pod
        if self.settings.pod is not None:
            settings_ = {'tolerance': self.settings.pod.tolerance,
                         'dim_max': self.settings.pod.dim_max,
                         'corners': self.settings.space.corners}
            self.pod = Pod(**settings_)
        else:
            self.pod = None
            self.logger.info('No POD is computed.')

        self.data = None
        # Surrogate model
        if self.settings.surrogate is not None:
            settings_ = {'kind': self.settings.surrogate.method,
                         'corners': self.settings.space.corners,
                         'plabels': self.settings.snapshot.plabels}
            if self.settings.surrogate.method == 'pc':
                dists = self.settings.space.sampling.distributions
                dists = dists_to_ot(dists)

                settings_.update({
                    'strategy': self.settings.surrogate.strategy,
                    'degree': self.settings.surrogate.degree,
                    'distributions': dists,
                    'sparse_param': self.settings.surrogate.sparse_param,
                    'sample': self.space.values
                })
            elif self.settings.surrogate.method == 'evofusion':
                settings_.update({
                    'cost_ratio': self.settings.surrogate.cost_ratio,
                    'grand_cost': self.settings.surrogate.grand_cost
                })
            elif self.settings.surrogate.method == 'kriging':
                if self.settings.surrogate.kernel is not None:
                    kernel = self.settings.surrogate.kernel
                    try:
                        kernel = eval(kernel, {'__builtins__': None},
                                      kernels.__dict__)
                    except (TypeError, AttributeError):
                        self.logger.error('Scikit-Learn kernel unknown.')
                        raise SystemError
                    settings_.update({'kernel': kernel})

                settings_.update({
                    'noise': self.settings.surrogate.noise,
                    'global_optimizer': self.settings.surrogate.global_optimizer
                })
            elif self.settings.surrogate.method == 'mixture':
                self.pod = None

                if self.settings.pod is not None:
                    pod_args = {'tolerance': self.settings.pod.tolerance,
                                'dim_max': self.settings.pod.dim_max}
                else:
                    pod_args = None

                settings_.update({
                    'pod': pod_args,
                    'plabels': self.settings.snapshot.plabels,
                    'corners': self.settings.space.corners,
                    'fsizes': self.settings.snapshot.fsizes[0],
                    'pca_percentage': self.settings.surrogate.pca_percentage,
                    'clusterer': self.settings.surrogate.clusterer,
                    'classifier': self.settings.surrogate.classifier
                })

            self.surrogate = SurrogateModel(**settings_)
            if self.settings.surrogate.method == 'pc':
                self.space.empty()
                self.space += self.surrogate.predictor.sample
                self.to_compute_points = self.space.values
        else:
            self.surrogate = None
            self.logger.info('No surrogate is computed.')

    def sampling(self, points=None, update=False):
        """Create snapshots.

        Generates or retrieve the snapshots [and then perform the POD].

        :param :class:`Space` points: points to perform the sample from.
        :param bool update: perform dynamic or static computation.
        """
        if points is None:
            points = self.to_compute_points

        # Generate snapshots
        samples = self.provider.require_data(points)
        self.snapshot_counter += len(samples)  # still useless

        # Fit the Surrogate [and POD]
        if self.pod is not None:
            if update:
                self.surrogate.space.empty()
                self.pod.update(samples)
            else:
                self.pod.fit(samples)
            self.data = self.pod.VS
            points = self.pod.space

        else:
            # [TODO] Über complicated pour rien ! --> révision du space + data
            if len(samples) > 0:
                data = samples.data
                if update:
                    data = np.append(self.data, data, axis=0)
                    if len(data) > len(self.space):  # no resampling
                        self.space += samples.space
                self.data = data
            points = self.space.values
        try:  # if surrogate
            self.surrogate.fit(points, self.data, pod=self.pod)
        except AttributeError:
            pass

    def resampling(self):
        """Resampling of the parameter space.

        Generate new samples if quality and number of sample are not satisfied.
        From a new sample, it re-generates the POD.

        """
        self.logger.info("\n----- Resampling parameter space -----")
        method = self.settings.space.resampling.method
        extremum = self.settings.space.resampling.extremum
        hybrid = self.settings.space.resampling.hybrid
        discrete = self.settings.space.sampling.discrete
        delta_space = self.settings.space.resampling.delta_space
        q2_criteria = self.settings.space.resampling.q2_criteria
        pdf = self.settings.uq.pdf if self.settings.uq is not None else None

        while len(self.space) < self.space.max_points_nb:
            self.logger.info("-> New iteration")

            if (method != 'optimization') and (q2_criteria is not None):
                quality, point_loo = self.surrogate.estimate_quality()
                if quality >= q2_criteria:
                    break
            elif 'loo' in method:
                _, point_loo = self.surrogate.estimate_quality()
            else:
                point_loo = None

            new_point = self.space.refine(self.surrogate,
                                          method,
                                          point_loo=point_loo,
                                          delta_space=delta_space, dists=pdf,
                                          hybrid=hybrid, discrete=discrete,
                                          extremum=extremum)

            try:
                self.sampling(new_point, update=True)
            except ValueError:
                break

            if method == 'optimization':
                self.space.optimization_results(extremum=extremum)

    def write(self):
        """Write DOE, Surrogate [and POD] to disk."""
        path = os.path.join(self.fname, self.fname_tree['space'])
        os.makedirs(path, exist_ok=True)

        self.space.write(path, 'space.dat')
        if self.surrogate is not None:
            path = os.path.join(self.fname, self.fname_tree['surrogate'])
            os.makedirs(path, exist_ok=True)
            self.surrogate.write(path)
        if self.pod is not None:
            path = os.path.join(self.fname, self.fname_tree['pod'])
            os.makedirs(path, exist_ok=True)
            self.pod.write(path)
        elif (self.pod is None) and (self.surrogate is None):
            path = os.path.join(self.fname, self.fname_tree['data'])
            with open(path, 'wb') as fdata:
                pickler = pickle.Pickler(fdata)
                pickler.dump(self.data)
            self.logger.debug('Wrote data to {}'.format(path))

    def read(self):
        """Read Surrogate [and POD] from disk."""
        path = os.path.join(self.fname, self.fname_tree['space'])
        self.space.empty()
        self.space.read(os.path.join(path, 'space.dat'))
        if self.surrogate is not None:
            self.surrogate.read(os.path.join(self.fname,
                                             self.fname_tree['surrogate']))
            self.data = self.surrogate.data
        if self.pod is not None:
            self.pod.read(os.path.join(self.fname, self.fname_tree['pod']))
            self.surrogate.pod = self.pod
        elif (self.pod is None) and (self.surrogate is None):
            path = os.path.join(self.fname, self.fname_tree['data'])
            with open(path, 'rb') as fdata:
                unpickler = pickle.Unpickler(fdata)
                self.data = unpickler.load()
            self.logger.debug('Data read from {}'.format(path))

    def restart(self):
        """Restart process."""
        # Surrogate [and POD] has already been computed
        self.logger.info('Restarting from previous computation...')
        to_compute_points = self.space.values
        self.read()  # Reset space with actual computations
        self.snapshot_counter = len(self.space)

        if self.snapshot_counter < len(to_compute_points):
            # will add new points to be processed
            # [static or dynamic pod is finished]
            self.to_compute_points = [p for p in to_compute_points
                                      if p not in self.space]
        else:
            # automatic resampling has to continue from
            # the processed points
            self.to_compute_points = []

    def prediction(self, points, write=False):
        """Perform a prediction.

        :param points: point(s) to predict.
        :type points: :class:`space.point.Point` or array_like
          (n_samples, n_features).
        :param bool write: whether to write snapshots.
        :return: Result.
        :rtype: array_like (n_samples, n_features).
        :return: Standard deviation.
        :rtype: array_like (n_samples, n_features).
        """
        results, sigma = self.surrogate(points)
        if write:
            path = os.path.join(self.fname, self.fname_tree['predictions'])
            space_fname = os.path.join(path, 'sample-space.json')
            data_fname = os.path.join(path, 'sample-data.json')
            os.makedirs(path, exist_ok=True)
            # better: surrogate could return a Sample
            plabels = self.settings.snapshot.plabels
            if self.space.multifidelity:
                plabels = plabels[1:]

            samples = Sample(space=points, data=results,
                             plabels=plabels,
                             flabels=self.settings.snapshot.flabels,
                             psizes=self.settings.snapshot.psizes,
                             fsizes=self.settings.snapshot.fsizes)
            samples.write(space_fname, data_fname)

        return results, sigma

    def uq(self):
        """Perform UQ analysis."""
        args = {}
        args['fname'] = os.path.join(self.fname, self.fname_tree['uq'])
        args['space'] = self.space
        args['indices'] = self.settings.uq.type
        args['plabels'] = self.settings.snapshot.plabels
        args['dists'] = self.settings.uq.pdf
        args['nsample'] = self.settings.uq.sample

        if self.space.multifidelity:
            args['plabels'] = args['plabels'][1:]

        if self.pod is not None:
            args['data'] = self.pod.inverse_transform(self.data)
        else:
            args['data'] = self.data

        args['test'] = self.settings.uq.test

        if self.settings.visualization is not None:
            args['xdata'] = self.settings.visualization.xdata
            args['xlabel'] = self.settings.visualization.xlabel
            args['flabel'] = self.settings.visualization.flabel

        mesh = None
        if (self.settings.visualization is not None) and \
                (self.settings.visualization.mesh_2D is not None):
            mesh = self.settings.visualization.mesh_2D.dict()

        analyse = UQ(self.surrogate, mesh=mesh, **args)

        if self.surrogate is None:
            self.logger.warning("No surrogate model, be sure to have a "
                                "statistically significant sample to trust "
                                "following results.")
        if len(self.settings.space.corners[0]) > 1:
            analyse.sobol()
        analyse.error_propagation()

    def visualization(self):
        """Apply visualisation options."""
        p_len = len(self.settings.space.corners[0])

        # In case of POD, data need to be converted from modes to snapshots.
        if self.pod is not None:
            data = self.pod.inverse_transform(self.data)
        else:
            data = self.data

        output_len = np.asarray(data).shape[1]

        self.logger.info('Creating response surface...')
        args = {}

        path = os.path.join(self.fname, self.fname_tree['visualization'])
        try:
            os.makedirs(path)
        except OSError:
            pass

        if self.settings.visualization is not None:
            # xdata for output with dim > 1
            if (self.settings.visualization.xdata is not None) and (output_len > 1):
                args['xdata'] = self.settings.visualization.xdata
            elif output_len > 1:
                args['xdata'] = np.linspace(0, 1, output_len)

            # Plot Doe if doe option is True
            if self.settings.visualization.doe:
                args['doe'] = self.space

            # Display resampling if resampling option is true
            if self.settings.visualization.resampling:
                args['resampling'] = self.settings.space.resampling.resamp_size
            else:
                args['resampling'] = 0

            args['ticks_nbr'] = self.settings.visualization.ticks_nbr
            args['contours'] = self.settings.visualization.contours
            args['range_cbar'] = self.settings.visualization.range_cbar
            args['axis_disc'] = self.settings.visualization.axis_disc

            # Create the 2D mesh graph
            if self.settings.visualization.mesh_2D is not None:
                self.logger.info("Creating 2D statistic graph from mesh...")
                output_path = os.path.join(path, 'Mesh_graph.pdf')
                mesh_2D(output_path=output_path, **self.settings.visualization.mesh_2D.dict())

        else:
            args['xdata'] = np.linspace(0, 1, output_len) if output_len > 1 else None

        try:
            args['bounds'] = self.settings.visualization.bounds
            for i, _ in enumerate(args['bounds'][0]):
                if (args['bounds'][0][i] < self.settings.space.corners[0][i]) \
                        or (args['bounds'][1][i] > self.settings.space.corners[1][i]):
                    args['bounds'] = self.settings.space.corners
                    self.logger.warning("Specified bounds for visualisation are "
                                        "wider than space corners. Default value used.")
        except KeyError:
            args['bounds'] = self.settings.space.corners

        # Data based on surrogate model (function) or not
        if 'surrogate' in self.settings:
            args['fun'] = self.func
        else:
            args['sample'] = self.space
            args['data'] = data

        try:
            args['plabels'] = self.settings.visualization.plabels
        except KeyError:
            args['plabels'] = self.settings.snapshot.plabels
        finally:
            if self.space.multifidelity:
                args['plabels'] = args['plabels'][1:]

        if len(self.settings.snapshot.flabels) < 2:
            try:
                args['flabel'] = self.settings.visualization.flabel
            except KeyError:
                args['flabel'] = self.settings.snapshot.flabels[0]

        if p_len < 5:
            # Creation of the response surface(s)
            args['fname'] = os.path.join(path, 'Response_Surface')
            response_surface(**args)

        else:
            # Creation of the Kiviat image
            args['fname'] = os.path.join(path, 'Kiviat.pdf')
            args['sample'] = self.space
            args['data'] = data
            if 'range_cbar' not in args:
                args['range_cbar'] = None
            if 'ticks_nbr' not in args:
                args['ticks_nbr'] = 10
            if 'kiviat_fill' not in args:
                args['kiviat_fill'] = True
            kiviat = Kiviat3D(args['sample'], data=args['data'],
                              bounds=args['bounds'], plabels=args['plabels'],
                              range_cbar=args['range_cbar'])
            kiviat.plot(fname=args['fname'], flabel=args['flabel'],
                        ticks_nbr=args['ticks_nbr'], fill=args['kiviat_fill'])

            # Creation of the Kiviat movie:
            args['fname'] = os.path.join(path, 'Kiviat.mp4')
            rate = 400
            kiviat.f_hops(frame_rate=rate, fname=args['fname'],
                          flabel=args['flabel'], fill=args['kiviat_fill'],
                          ticks_nbr=args['ticks_nbr'])

    @multi_eval
    def func(self, coords):
        """Evaluate the surrogate at a given point.

        This function calls the surrogate to compute a prediction.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float.
        """
        f_eval, _ = self.surrogate(coords)
        return f_eval[0]
