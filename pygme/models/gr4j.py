
import numpy as np
import pandas as pd

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_gr4j
import c_pygme_models_utils

# Dimensions
NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()


class GR4J(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        self._nuh1 = 0
        self._nuh2 = 0

        Model.__init__(self, 'gr4j',
            nconfig=1,
            ninputs=2,
            nparams=4,
            nstates=2,
            noutputs_max=9,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = 'catcharea'

        self._params.names = ['S', 'IGF', 'R', 'TB']
        self._params.min = [10., -50., 1., 0.5]
        self._params.max = [20000., 50., 5000., 100.]
        self._params.default = [400., -1., 50., 0.5]


    def post_params_setter(self):

        super(GR4J, self).post_params_setter()

        # Get parameters
        params = self.params

        # Set first uh
        nuh1 = np.zeros(1).astype(np.int32)
        uh1 = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_pygme_models_utils.uh_getuh(NUHMAXLENGTH,
                1, params[3], \
                nuh1, uh1)
        self._nuh1 = nuh1[0]

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='Model GR4J: c_pygme_models_utils.uh_getuh')

        self.uh[:self._nuh1] = uh1[:self._nuh1]

        # Set second uh
        nuh2 = np.zeros(1).astype(np.int32)
        uh2 = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_pygme_models_utils.uh_getuh(NUHMAXLENGTH, \
                2, params[3], \
                nuh2, uh2)
        self._nuh2 = nuh2[0]

        if ierr > 0:
            raise ValueError(('Model GR4J: c_pygme_models_utils.uh_getuh' +
                ' returns {0}').format(ierr))

        nend = self._uh.nval-self._nuh1
        self.uh[self._nuh1:] = uh2[:nend]
        self._nuhlength = self._nuh1 + self._nuh2

        # Set default initial states
        if self._states is None:
            raise ValueError(('{0} model: states are None,' +
                    ' please allocate').format(self.name))

        self._states.default = [params[0] * 0.5, params[2] * 0.4]


    def run(self, seed=None):

        start, end = self.startend

        ierr = c_pygme_models_gr4j.gr4j_run(self._nuh1,
            self._nuh2, start, end,
            self._params.data,
            self._uh.data,
            self._uh.data[self._nuh1:],
            self._inputs.data,
            self._statesuh.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_gr4j.gr4j_run' +
                ' returns {0}').format(ierr))


    def runtimestep(self, index, seed=None):

        ierr = c_pygme_models_gr4j.gr4j_run(self._nuh1,
            self._nuh2, start, end,
            self._params.data,
            self._uh.data,
            self._uh.data[self._nuh1:],
            self._inputs.data,
            self._statesuh.data,
            self._states.data,
            self._outputs.data)


class CalibrationGR4J(Calibration):

    def __init__(self, timeit=False):

        gr = GR4J()

        Calibration.__init__(self,
            model = gr, \
            ncalparams = 4, \
            timeit = timeit)

        self._calparams.means =  [5.8, -0.78, 3.39, 0.86]

        covar = [[1.16, 0.2, -0.15, -0.07],
                [0.2, 1.79, -0.24, -0.149],
                [-0.15, -0.24, 1.68, -0.16],
                [-0.07, -0.149, -0.16, 0.167]]
        self._calparams.covar = covar


    def cal2true(self, calparams):
        params = np.array([np.exp(calparams[0]),
                np.sinh(calparams[1]),
                np.exp(calparams[2]),
                0.49+np.exp(calparams[3])])

        return params


