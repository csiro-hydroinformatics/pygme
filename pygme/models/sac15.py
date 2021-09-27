
import numpy as np
import pandas as pd

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_hydromodels
import c_pygme_models_utils

# Dimensions
NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()


class SAC15(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        self._nuh = 0

        Model.__init__(self, 'sac15',
            nconfig=1,
            ninputs=2,
            nparams=15,
            nstates=6,
            noutputs_max=6,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = 'catcharea'

        self._params.names = ['Adimp', 'Lzfpm', 'Lzfsm', 'Lzpk',
                    'Lzsk', 'Lztwm', 'Pfree', 'Rexp', 'Sarva',
                    'Side', 'Lag', 'Uzfwm', 'Uzk', 'Uztwm', 'Zperc']

        self._params.min = [1e-5, 1e-2, 1e-2, 1e-3, 1e-3, 10.,
                1e-2, 1., 0., -0.2, 0, 1e-1, 1e-5, 1., 1e-2]

        self._params.max = [0.9, 5e2, 5e2, 0.9, 0.9, 1e3, 0.5,
                6., 0.5, 0.5, 100., 1e3, 1-1e-10, 1e3, 1e3]

        self._params.default = [0.1, 82., 32., 0.04, 0.24, 179.,
                0.4, 1.81, 0.01, 0., 0., 49., 0.4, 76., 60.]


    def post_params_setter(self):

        super(SAC15, self).post_params_setter()

        # Get parameters
        params = self.params

        # Set uh (uhid = 5 -> lag)
        # if uhid = 6 -> triangle
        nuh = np.zeros(1).astype(np.int32)
        uh = np.zeros(NUHMAXLENGTH).astype(np.float64)
        Lag = params[10]
        ierr = c_pygme_models_utils.uh_getuh(NUHMAXLENGTH,
                5, Lag, nuh, uh)
        self._nuh = nuh[0]

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='Model SAC15: c_pygme_models_utils.uh_getuh')

        self.uh[:self._nuh] = uh[:self._nuh]

        # Set default initial states
        if self._states is None:
            raise ValueError(('{0} model: states are None,' +
                    ' please allocate').format(self.name))

        # Initialise stores
        Lzfpm = params[1]
        Lzfsm = params[2]
        Lztwm = params[5]
        Uzfwm = params[11]
        Uztwm = params[13]
        # -> [Uztwc, Uzfwc, Lztwc, Lzfsc, Lzfpc, Adimc]
        self._states.default = [Uztwm, Uzfwm/2, Lztwm/2,
                Lzfsm/2, Lzfpm/2, Uztwm+Lztwm/2]


    def runblock(self, istart, iend, seed=None):

        ierr = c_pygme_models_hydromodels.sac15_run(self._nuh,
            istart, iend,
            self._params.data,
            self._uh.data,
            self._inputs.data,
            self._statesuh.data,
            self._states.data,
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.sac15_run' +
                ' returns {0}').format(ierr))



class CalibrationSAC15(Calibration):

    def __init__(self, timeit=False):

        sa = SAC15()

        Calibration.__init__(self,
            model = sa,
            warmup = 365,
            timeit = timeit)

        # Calibration on sse square root with bias constraint
        self._errfun.constants = [0.5, 2., 1.]

        self._calparams.means =  [5.8, -0.78, 3.39, 0.86]

        covar = [[1.16, 0.2, -0.15, -0.07],
                [0.2, 1.79, -0.24, -0.149],
                [-0.15, -0.24, 1.68, -0.16],
                [-0.07, -0.149, -0.16, 0.167]]
        self._calparams.covar = covar


    def cal2true(self, calparams):

        params = np.zeros(len(calparams), np.float64)

        params[0]  = calparams[0] #   ADIMP
        params[1]  = math.exp(calparams[1]) #   LZFPM
        params[2]  = math.exp(calparams[2]) #   LZFSM
        params[3]  = 1./(1.+math.exp(-calparams[3])) #   LZPK
        params[4]  = 1./(1.+math.exp(-calparams[4])) #   LZSK
        params[5]  = math.exp(calparams[5]) #   LZTWM
        params[6]  = (9.99+calparams[6])/19.98 #   PFREE
        params[7]  = math.exp(calparams[7]) #   REXP
        params[8]  = calparams[8] #   SARVA
        params[9]  = math.sinh((5.+calparams[9])/10.) #   SIDE
        params[10] = math.exp(calparams[10]) #   LAG
        params[11] = math.exp(calparams[11]) #   UZFWM
        params[12] = 1./(1.+math.exp(-calparams[12])) #   UZK
        params[13] = math.exp(calparams[13]) #   UZTWM
        params[14] = math.exp(calparams[14]) #   ZPERC

        return params


