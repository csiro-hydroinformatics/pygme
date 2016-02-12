
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_knn


class KNN(Model):

    def __init__(self,
            knn_var,
            knn_weights=None,
            knn_cycle_position=0,
            out_var=None,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        self.seed = 333

        # Set default variables
        if out_var is None:
            out_var = knn_var

        # Check inputs
        _knn_var = np.atleast_2d(knn_var)
        if _knn_var.shape[0] == 1:
            _knn_var = _knn_var.T

        _out_var = np.atleast_2d(out_var)
        if _out_var.shape[0] == 1:
            _out_var = _out_var.T

        if knn_weights is None:
            _knn_weights = np.ones(_knn_var.shape[0])
        else:
            _knn_weights = np.atleast_1d(knn_weights)

        if _knn_weights.shape[0] != _knn_var.shape[0]:
            raise ValueError(('KNN model: weights.shape[0]({0}) '+
                '!= knn_var.shape[0]({1})').format(_knn_weights.shape[0],
                    _knn_var.shape[0]))

        if _out_var.shape[0] != _knn_var.shape[0]:
            raise ValueError(('KNN model: out_var.shape[0]({0}) '+
                '!= knn_var.shape[0]({1})').format(_knn_weights.shape[0],
                    _knn_var.shape[0]))

        # Store special variables
        self.knn_cycle_position = np.int32(knn_cycle_position)
        self.knn_var = np.ascontiguousarray(_knn_var)
        self.out_var = _out_var
        self.knn_weights = _knn_weights
        self.idx_knn = None

        Model.__init__(self, 'knn',
            nconfig=0,
            ninputs=1,
            nparams=3,
            nstates=1,
            noutputs_max=self.out_var.shape[1],
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self._params.names = ['knn_halfwindow', 'knn_nb', 'cycle_length']
        self._params.min = [4, 3, 10]
        self._params.max = [50, 50, 366]
        self._params.default = [20, 10, 365.25]

        self.reset()


    def initialise(self, states=None, statesuh=None,
        cycle_position=None, knn_var=None):

        params = self.params

        if self._states is None:
            raise ValueError(('{0} model: states are None,' +
                    ' please allocate').format(self.name))

        # initialise KNN with random
        if states is None:

            statesuh = np.zeros(self._statesuh.nval)

            if not cycle_position is None and not knn_var is None:

                states = np.zeros(self._states.nval)

                knn_idx = np.zeros(1, dtype=np.int32)
                knn_var = np.atleast_1d(knn_var).astype(np.float64)

                ierr = c_pygme_models_knn.knn_mindist(cycle_position,
                    self.knn_cycle_position,
                    knn_var,
                    self.params,
                    self.knn_weights,
                    self.knn_var,
                    knn_idx)

                states[0] = knn_idx[0]

            else:
                states = np.zeros(self._states.nval)

        super(KNN, self).initialise(states, statesuh)


    def run(self):

        # Starting day
        idx_select = np.int32(self.states[0])

        # outputs
        nval, _, _, _ = self.get_dims('outputs')
        self.knn_idx = np.zeros(nval, dtype=np.int32)

        ierr = c_pygme_models_knn.knn_run(idx_select,
            self.seed,
            self.params,
            self.knn_weights,
            self.knn_var,
            self.knn_idx)

        if ierr > 0:
            raise ValueError(('c_pygme_models_knn.knn_run' +
                ' returns {0}').format(ierr))

        # Save resampled data
        _, noutputs, _, _ = self.get_dims('outputs')
        self.outputs = self.out_var[self.knn_idx, :noutputs]

