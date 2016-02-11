
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_knn


class KNN(Model):

    def __init__(self,
            var_knn, weights_knn,
            var_out=None,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        # Output variables
        if var_out is None:
            var_out = var_knn

        # Check inputs
        _var_knn = np.atleast_2d(var_knn)
        _var_out = np.atleast_2d(var_out)
        _weights = np.atleast_1d(weights_knn)

        if _weights.shape[0] != _var_knn.shape[0]:
            raise ValueError(('KNN model: weights.shape[0]({0}) '+
                '!= var_knn.shape[0]({1})').format(_weights.shape[0],
                    _var_knn.shape[0]))

        if _var_out.shape[0] != _var_knn.shape[0]:
            raise ValueError(('KNN model: var_out.shape[0]({0}) '+
                '!= var_knn.shape[0]({1})').format(_weights.shape[0],
                    _var_knn.shape[0]))

        # Store special variables
        self.var_knn = _var_knn
        self.var_out = _var_out
        self.weights_knn = _weights
        self.idx_knn = None

        Model.__init__(self, 'knn',
            nconfig=0,
            ninputs=1,
            nparams=3,
            nstates=1,
            noutputs_max=self.var_out.shape[1],
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self._params.names = ['knn_window', 'knn_nb', 'cycle_length']
        self._params.min = [5, 2, 10]
        self._params.max = [50, 20, 366]
        self._params.default = [20, 6, 365.25]

        self.reset()


    def run(self):

        if self._inputs.nvar != self.ninputs:
            raise ValueError(('Model KNN, self._inputs.nvar({0}) != ' +
                    'self._ninputs({1})').format(
                    self._inputs.nvar, self.ninputs))

        # Starting day
        idx_select = np.int32(self.states[0])

        # outputs
        nval, _, _, _ = self.get_dims('outputs')
        self.knn_idx = np.zeros(nval, dtype=np.int32)

        ierr = c_pygme_models_knn.knn_run(idx_select,
            self.params,
            self.weights_knn,
            self.var_knn,
            self.inputs[:, 0],
            self.knn_idx)

        if ierr > 0:
            raise ValueError(('c_pygme_models_knn.knn_run' +
                ' returns {0}').format(ierr))

        # Save resampled data
        _, noutputs, _, _ = self.get_dims('outputs')
        self.outputs = self.var_out[self.knn_idx, :noutputs]

