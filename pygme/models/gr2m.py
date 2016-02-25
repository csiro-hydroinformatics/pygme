
import math
import numpy as np
import pandas as pd

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_gr2m
import c_pygme_models_utils


class GR2M(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):


        Model.__init__(self, 'gr2m',
            nconfig=1,
            ninputs=2,
            nparams=2,
            nstates=2,
            noutputs_max=9,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = 'catcharea'

        self._params.names = ['S', 'IGF']
        self._params.min = [10., 0.1]
        self._params.max = [10000., 3.]
        self._params.default = [400., 0.8]

        self.reset()


    def initialise(self, states=None, statesuh=None):

        params = self.params

        if self._states is None:
            raise ValueError(('{0} model: states are None,' +
                    ' please allocate').format(self.name))

        # initialise GR4J with reservoir levels
        if states is None:
            states = np.zeros(self._states.nval)
            states[0] = params[0] * 0.5
            states[1] = params[1] * 0.4

            statesuh = np.zeros(self._statesuh.nval)

        super(GR2M, self).initialise(states, statesuh)



    def run(self, seed=None):

        start, end = self.startend

        ierr = c_pygme_models_gr2m.gr2m_run(start, end,
            self._params.data, \
            self._inputs.data, \
            self._states.data, \
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('Model gr2m, c_pygme_models_gr2m.gr2m_run' + \
                    'returns {0}').format(ierr))



class CalibrationGR2M(Calibration):

    def __init__(self, timeit=False):

        gr = GR2M()

        Calibration.__init__(self,
            model = gr, \
            ncalparams = 2, \
            timeit = timeit)

        self._calparams.means =  [5.9, -0.28]
        self._calparams.covar = [[0.52, 0.015], [0.015, 0.067]]


    def cal2true(self, calparams):
        return np.exp(calparams)


