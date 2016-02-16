
import numpy as np
import pandas as pd

from hystat import sutils

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_lagroute
import c_pygme_models_utils

# Dimensions
NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()


class LagRoute(Model):

    def __init__(self,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):


        Model.__init__(self, 'lagroute',
            nconfig=4,
            ninputs=1,
            nparams=2,
            nstates=1,
            noutputs_max = 4,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['timestep', 'length', \
                'flowref', 'storage_expon']

        self.config['timestep'] = 86400
        self.config['length'] = 86400
        self.config['flowref'] = 1
        self.config['storage_expon'] = 1

        self._params.names = ['U', 'alpha']
        self._params.min = [0.01, 0.]
        self._params.max = [20., 1.]
        self._params.default = [1., 0.5]

        self.reset()


    def set_uh(self):

        # Lag = alpha * U * L / dt
        config = self.config
        params = self._params

        delta = config['length'] * params['U'] * params['alpha']
        delta /= config['timestep']
        delta = np.float64(delta)

        if np.isnan(delta):
            raise ValueError(('Problem with delta calculation. ' + \
                'One of config[\'length\']{0}, config[\'timestep\']{1}, ' + \
                'params[\'U\']{2} or params[\'alpha\']{3} is NaN').format( \
                config['length'], config['timestep'], params['U'],
                params['alpha']))

        # First uh
        nuh = np.zeros(1).astype(np.int32)
        uh = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_pygme_models_utils.uh_getuh(NUHMAXLENGTH,
                5, delta, \
                nuh, uh)

        if ierr > 0:
            raise ValueError(('Model LagRoute: c_pygme_models_utils.uh_getuh' + \
                ' returns {0}').format(ierr))

        self._uh.data = uh
        self._nuhlength = nuh[0]


    def run(self):

        _, ninputs, _, _ = self.get_dims('inputs')
        if self._inputs.nvar != ninputs:
            raise ValueError(('Model LagRoute, self.inputs.nvar({0}) != ' + \
                    'self.ninputs({1})').format( \
                    self._inputs.nvar, ninputs))

        ierr = c_pygme_models_lagroute.lagroute_run(self._nuhlength, \
            self.config.data, \
            self._params.data, \
            self._uh.data, \
            self._inputs.data, \
            self._statesuh.data, \
            self._states.data, \
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_lagroute.' + \
                'lagroute_run returns {0}').format(ierr))


class CalibrationLagRoute(Calibration):

    def __init__(self, timeit=False):

        lm = LagRoute()

        Calibration.__init__(self,
            model = lm, \
            ncalparams = 2, \
            timeit = timeit)

        self._calparams.means =  [1., 0.5]

        covar = [[0.5, 0.], [0., 0.2]]
        self._calparams.covar = covar


