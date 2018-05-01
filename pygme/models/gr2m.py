
import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels


class GR2M(Model):

    def __init__(self):

        # Config vector
        config = Vector(['catcharea'], [0], [0])

        # params vector
        vect = Vector(['X1', 'X2'], \
                    [400, 0.8], [10., 0.1], [1e4, 3])
        params = ParamsVector(vect)

        # State vector
        states = Vector(['S', 'R'])

        # Model
        super(GR2M, self).__init__('GR2M',
            config, params, states, \
            ninputs=2, \
            noutputsmax=9)

        self.outputs_names = ['Q', 'F', 'P1', 'P2', 'P3', \
                            'R1', 'R2', 'S', 'R']


    def initialise_fromdata(self):
        ''' Initialisation of GR2M using
            * Production store: 50% filling level
            * Routing store: 30% filling level
        '''
        X1 = self.params.X1

        # Production store
        S0 = 0.5 * X1

        # Routing store 60x30% = 18 mm
        R0 = 18

        # Model initialisation
        self.initialise(states=[S0, R0])


    def run(self):
        ierr = c_pygme_models_hydromodels.gr2m_run(self.istart, self.iend,
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(('Model gr2m, c_pygme_models_hydromodels.gr2m_run' + \
                    'returns {0}').format(ierr))



class CalibrationGR2M(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
            warmup=36, \
            timeit=False,\
            fixed=None, \
            objfun_kwargs={}):

        # Input objects for Calibration class
        model = GR2M()
        params = model.params

        cp = Vector(['tX1', 'tX2'], \
                mins=np.log(params.mins),
                maxs=np.log(params.maxs),
                defaults=np.log(params.defaults))
        calparams = CalibParamsVector(model, cp, \
            trans2true='exp', fixed=fixed)

        plib = np.random.multivariate_normal(mean=params.defaults, \
                    cov=np.diag((params.maxs-params.mins)/2), \
                    size=500)
        plib = np.clip(plib, params.mins, params.maxs)

        # Initialisation arguments
        initial_kwargs = {}

        # Instanciate calibration
        super(CalibrationGR2M, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib, \
            objfun_kwargs=objfun_kwargs, \
            initial_kwargs=initial_kwargs)



