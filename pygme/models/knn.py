
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_knn


def dayofyear(d):
    ''' Return the day of the year for a date-time object '''
    return d.timetuple().tm_yday


class KNN(Model):

    def __init__(self,
            input_var,
            input_weights=None,
            knn_cycle_position=0,
            output_var=None,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        # Set default variables
        if output_var is None:
            output_var = input_var

        # Check inputs
        _input_var = np.atleast_2d(input_var).astype(np.float64)
        if _input_var.shape[0] == 1:
            _input_var = _input_var.T

        _output_var = np.atleast_2d(output_var).astype(np.float64)
        if _output_var.shape[0] == 1:
            _output_var = _output_var.T

        if input_weights is None:
            _input_weights = np.ones(_input_var.shape[0], dtype=np.float64)
        else:
            _input_weights = np.atleast_1d(input_weights).astype(np.float64)

        if _input_weights.shape[0] != _input_var.shape[0]:
            raise ValueError(('KNN model: weights.shape[0]({0}) '+
                '!= input_var.shape[0]({1})').format(_input_weights.shape[0],
                    _input_var.shape[0]))

        if _output_var.shape[0] != _input_var.shape[0]:
            raise ValueError(('KNN model: output_var.shape[0]({0}) '+
                '!= input_var.shape[0]({1})').format(_output_var.shape[0],
                    _input_var.shape[0]))

        # Store special variables
        self.knn_cycle_position = np.int32(knn_cycle_position)
        self.input_var = np.ascontiguousarray(_input_var)
        self.output_var = _output_var
        self.input_weights = _input_weights
        self.idx_knn = None

        Model.__init__(self, 'knn',
            nconfig=5,
            ninputs=1,
            nparams=0,
            nstates=_input_var.shape[1] + 1,
            noutputs_max=self.output_var.shape[1],
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['halfwindow', 'nb_nn',
                                'cycle_length',
                                'cycle_position_ini',
                                'cycle_position_ini_opt']
        self.config.min = [4, 3, 10, 0, 0]
        self.config.max = [50, 50, 366, 366, 2]
        self.config.default = [20, 10, 365.25, 0, 0]
        self.config.reset()


    def runblock(self, istart, iend, seed=None):

        # Seed
        if seed is None:
            seed = np.int32(-1)

        # outputs
        nval, _, _, _ = self.get_dims('outputs')
        self.knn_idx = np.zeros(nval, dtype=np.int32)

        # Run model
        ierr = c_pygme_models_knn.knn_run(seed, istart, iend,
            self.config.data,
            self.input_weights,
            self.input_var,
            self.states,
            self.knn_idx)

        if ierr > 0:
            raise ValueError(('c_pygme_models_knn.knn_run' +
                ' returns {0}').format(ierr))

        # Save resampled data
        _, noutputs, _, _ = self.get_dims('outputs')
        self.outputs = self.output_var[self.knn_idx, :noutputs]


