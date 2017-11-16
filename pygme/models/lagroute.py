
import numpy as np
import pandas as pd

from pygme.model import Model
from pygme.calibration import Calibration

import c_pygme_models_hydromodels
import c_pygme_models_utils

# Dimensions
NUHMAXLENGTH = c_pygme_models_utils.uh_getnuhmaxlength()


class LagRoute(Model):

    def __init__(self):

        # Config vector
        config = Vector(['timestep', 'length', \
                            'flowref', 'storage_expon'], \
                    [86400, 1e5, 1, 1], \
                    [0, 0, 0, 1e-6, 1e-2], \
                    [np.inf, np.inf, np.inf, 2])

        # params vector
        vect = Vector(['U', 'alpha'], \
                    [1., 0.5], \
                    [0.01, 0.], \
                    [1., 0.5])
        uhs = [UH('lag', 3)]
        params = ParamsVector(vect, uhs)

        # State vector
        states = Vector(['S'])

        # Model
        super(LagRoute, self).__init__('LagRoute',
            config, params, states, \
            ninputs=1, \
            noutputsmax=4)


    #def post_params_setter(self):

    #    # Lag = alpha * U * L / dt
    #    config = self.config
    #    params = self._params

    #    delta = config['length'] * params['U'] * params['alpha']
    #    delta /= config['timestep']
    #    delta = np.float64(delta)

    #    if np.isnan(delta):
    #        raise ValueError(('Problem with delta calculation. ' + \
    #            'One of config[\'length\']{0}, config[\'timestep\']{1}, ' + \
    #            'params[\'U\']{2} or params[\'alpha\']{3} is NaN').format( \
    #            config['length'], config['timestep'], params['U'],
    #            params['alpha']))

    #    # First uh
    #    nuh = np.zeros(1).astype(np.int32)
    #    uh = np.zeros(NUHMAXLENGTH).astype(np.float64)
    #    ierr = c_pygme_models_utils.uh_getuh(NUHMAXLENGTH,
    #            5, delta, \
    #            nuh, uh)

    #    if ierr > 0:
    #        raise ValueError(('Model LagRoute: c_pygme_models_utils.uh_getuh' + \
    #            ' returns {0}').format(ierr))

    #    self._uh.data = uh
    #    self._nuhlength = nuh[0]


    def run(self, istart, iend, seed=None):

        ierr = c_pygme_models_hydromodels.lagroute_run(self._nuhlength, \
            istart, iend,
            self.config.data, \
            self._params.data, \
            self._uh.data, \
            self._inputs.data, \
            self._statesuh.data, \
            self._states.data, \
            self._outputs.data)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.' + \
                'lagroute_run returns {0}').format(ierr))


class CalibrationLagRoute(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None):

        # Input objects for Calibration class
        model = LagRoute()
        params = model.params

        cp = Vector(['tU', 'talpha'], \
                mins=params.mins,
                maxs=params.maxs,
                defaults=params.defaults)

        # no parameter transformation
        calparams = CalibParamsVector(model, cp, \
            trans2true=trans2true,\
            true2trans=true2trans,\
            fixed=fixed)

        # Build parameter library from
        # systematic exploration of parameter space
        uu , aa = np.mesh_grid(np.linspace(0.1, 3, 20), \
                        np.linspace(0, 1, 20))
        plib = np.column_stack([uu.ravel(), aa.ravel()])

        # Instanciate calibration
        super(CalibrationLagRoute, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib)


