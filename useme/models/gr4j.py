
import numpy as np
import pandas as pd

from hystat import sutils

from useme.model import Model
from useme.calibration import Calibration

import c_useme_models_gr4j
import c_useme_models_utils

# Dimensions
NUHMAXLENGTH = c_useme_models_utils.uh_getnuhmaxlength()


class GR4J(Model):

    def __init__(self,
            nens_params=1,
            nens_states_random=1,
            nens_outputs_random=1):

        self._nuh1 = 0
        self._nuh2 = 0

        Model.__init__(self, 'gr4j',
            nconfig=1,
            ninputs=2,
            nparams=4,
            nstates=2,
            noutputs_max=9,
            nens_params=nens_params,
            nens_states_random=nens_states_random,
            nens_outputs_random=nens_outputs_random)

        self.config.names = 'catcharea'

        self._params.names = ['S', 'IGF', 'R', 'TB']
        self._params.min = [10., -50., 1., 0.5]
        self._params.max = [20000., 50., 5000., 100.]
        self._params.default = [400., -1., 50., 0.5]

        self.reset()


    def set_uh(self):

        super(GR4J, self).set_uh()

        params = self.params

        # First uh
        nuh1 = np.zeros(1).astype(np.int32)
        uh1 = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_useme_models_utils.uh_getuh(NUHMAXLENGTH,
                1, params[3], \
                nuh1, uh1)
        self._nuh1 = nuh1[0]

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='Model GR4J: c_useme_models_utils.uh_getuh')

        self.uh[:self._nuh1] = uh1[:self._nuh1]

        # Second uh
        nuh2 = np.zeros(1).astype(np.int32)
        uh2 = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_useme_models_utils.uh_getuh(NUHMAXLENGTH, \
                2, params[3], \
                nuh2, uh2)
        self._nuh2 = nuh2[0]

        if ierr > 0:
            raise ValueError(('Model GR4J: c_useme_models_utils.uh_getuh' +
                ' returns {0}').format(ierr))

        nend = self._uh.nval-self._nuh1
        self.uh[self._nuh1:] = uh2[:nend]
        self._nuhlength = self._nuh1 + self._nuh2


    def initialise(self, states=None, statesuh=None):

        params = self.params

        if self._states is None:
            raise ValueError(('{0} model: states are None,' +
                    ' please allocate').format(self.name))

        # initialise GR4J with reservoir levels
        if states is None:
            states = np.zeros(self._states.nval)
            states[0] = params[0] * 0.5
            states[1] = params[2] * 0.4

            statesuh = np.zeros(self._statesuh.nval)

        super(GR4J, self).initialise(states, statesuh)


    def run(self):

        if self._inputs.nvar != self.ninputs:
            raise ValueError(('Model GR4J, self._inputs.nvar({0}) != ' +
                    'self._ninputs({1})').format(
                    self._inputs.nvar, self.ninputs))

        ierr = c_useme_models_gr4j.gr4j_run(self._nuh1,
            self._nuh2,
            self._params.data,
            self._uh.data,
            self._uh.data[self._nuh1:],
            self._inputs.data,
            self._statesuh.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_useme_models_gr4j.gr4j_run' +
                ' returns {0}').format(ierr))


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


