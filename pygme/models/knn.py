
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_knn


class KNN(Model):

    def __init__(self, var, weights,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        self.var = var
        self.weights = weigths

        Model.__init__(self, 'knn',
            nconfig=0,
            ninputs=1,
            nparams=2,
            nstates=1,
            noutputs_max=1,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self._params.names = ['knn_window', 'knn_nb']
        self._params.min = [5, 2]
        self._params.max = [50, 20]
        self._params.default = [20, 6]

        self.reset()


    def run(self):

        if self._inputs.nvar != self.ninputs:
            raise ValueError(('Model KNN, self._inputs.nvar({0}) != ' +
                    'self._ninputs({1})').format(
                    self._inputs.nvar, self.ninputs))

        # Starting day
        idx_select = np.int32(states[0])

        # outputs
        nval, _, _, _ = self.get_dims('outputs')
        knn_idx = np.zeros(nval, dtypes=np.int32)

        ierr = c_pygme_models_knn.knn_run(idx_select,
            self.params,
            self.weights,
            self.var,
            self.inputs[:, 0],
            knn_idx)

        self.outputs[:, 0] = knn_idx

        if ierr > 0:
            raise ValueError(('c_pygme_models_knn.knn_run' +
                ' returns {0}').format(ierr))


