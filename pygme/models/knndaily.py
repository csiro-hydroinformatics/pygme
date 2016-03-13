
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.forecastmodel import ForecastModel

import c_pygme_models_knndaily


class KNNDaily(Model):

    def __init__(self,
            knnvar_inputs,
            knn_cycle_position=0,
            knnvar_outputs=None,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        # Set default variables
        if knnvar_outputs is None:
            knnvar_outputs = knnvar_inputs

        # Check inputs
        _knnvar_inputs = np.atleast_2d(knnvar_inputs).astype(np.float64)
        if _knnvar_inputs.shape[0] == 1:
            _knnvar_inputs = _knnvar_inputs.T

        _knnvar_outputs = np.atleast_2d(knnvar_outputs).astype(np.float64)
        if _knnvar_outputs.shape[0] == 1:
            _knnvar_outputs = _knnvar_outputs.T

        if _knnvar_outputs.shape[0] != _knnvar_inputs.shape[0]:
            raise ValueError(('KNNDAILY model: knnvar_outputs.shape[0]({0}) '+
                '!= knnvar_inputs.shape[0]({1})').format(_knnvar_outputs.shape[0],
                    _knnvar_inputs.shape[0]))

        # Store special variables
        self.knnvar_inputs = np.ascontiguousarray(_knnvar_inputs)
        self.knnvar_outputs = _knnvar_outputs
        self.ipos_knn = None

        Model.__init__(self, 'knndaily',
            nconfig=4,
            ninputs=1,
            nparams=0,
            nstates=_knnvar_inputs.shape[1] + 1,
            noutputs_max=self.knnvar_outputs.shape[1],
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['halfwindow', 'nb_nn',
                                'randomize_ties', 'date_ini']
        self.config.min = [4, 3, 0, 15000101.]
        self.config.max = [50, 50, 1, np.inf]
        self.config.default = [10, 5, 1, 20000101.]
        self.config.reset()


    def runblock(self, istart, iend, seed=None):

        # outputs
        nval, _, _, _ = self.get_dims('outputs')
        self.knn_ipos = np.zeros(nval, dtype=np.int32)

        # Run model
        ierr = c_pygme_models_knndaily.knndaily_run(istart, iend,
            self.config.data,
            self.inputs,
            self.knnvar_inputs,
            self.states,
            self.knn_ipos)

        if ierr > 0:
            raise ValueError(('c_pygme_models_knndaily.knndaily_run' +
                ' returns {0}').format(ierr))

        # Save resampled data
        _, noutputs, _, _ = self.get_dims('outputs')
        self.outputs = self.knnvar_outputs[self.knn_ipos, :noutputs]


