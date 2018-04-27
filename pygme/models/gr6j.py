import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector, UH
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels


class GR6J(Model):

    def __init__(self):

        # Config vector
        config = Vector(['continuous'],\
                    [0], [0], [1])

        # params vector
        vect = Vector(['X1', 'X2', 'X3', 'X4', 'X5', 'X6'], \
                    [400, -1, 50, 0.5, 0., 10.], \
                    [1, -50, 1, 0.5, -50., 1], \
                    [1e4, 50, 1e4, 1e2, 50., 1e5])
        params = ParamsVector(vect)

        # UH
        params.add_uh('gr4j_ss1_daily', lambda params: params.X4)
        params.add_uh('gr4j_ss2_daily', lambda params: params.X4)

        # State vector
        states = Vector(['S', 'R', 'A'])

        # Model
        super(GR6J, self).__init__('GR6J',
            config, params, states, \
            ninputs=2, \
            noutputsmax=11)

        self.outputs_names = ['Q', 'ECH', 'AE', \
                    'PR', 'QD', 'QR', 'PERC', 'QExp', 'S', 'R', 'A']


    def run(self):
        # Get uh object (not set_timebase function, see ParamsVector class)
        _, uh1 = self.params.uhs[0]
        _, uh2 = self.params.uhs[1]

        # Run gr4j c code
        ierr = c_pygme_models_hydromodels.gr6j_run(uh1.nord, \
            uh2.nord, self.istart, self.iend, \
            self.params.values, \
            uh1.ord, \
            uh2.ord, \
            self.inputs, \
            uh1.states, \
            uh2.states, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.gr6j_run' +
                ' returns {0}').format(ierr))


class CalibrationGR6J(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    objfun_kwargs={}):

        # Input objects for Calibration class
        model = GR6J()
        params = model.params


        trans2true = lambda x: np.array([
                        math.exp(x[0]), \
                        math.sinh(x[1]), \
                        math.exp(x[2]), \
                        0.49+math.exp(x[3]), \
                        math.sinh(x[4]), \
                        math.exp(x[5])
                    ])

        true2trans = lambda x: np.array([
                        math.log(x[0]), \
                        math.asinh(x[1]), \
                        math.log(x[2]), \
                        math.log(x[3]-0.49), \
                        math.asinh(x[4]), \
                        math.log(x[5])
                    ])

        cp = Vector(['tX1', 'tX2', 'tX3', 'tX4', 'tX5', 'tX6'], \
                mins=true2trans(params.mins),
                maxs=true2trans(params.maxs),
                defaults=true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=trans2true, \
            true2trans=true2trans,\
            fixed=fixed)

        # Build parameter library from
        # MVT norm in transform space
        tplib = np.random.multivariate_normal(\
                    mean=[5.8, -0.78, 3.39, 0.86, 0., 3.],\
                    cov = [[1.16, 0.2, -0.15, -0.07, 0., 0.],\
                            [0.2, 1.79, -0.24, -0.149, 0., 0.],\
                            [-0.15, -0.24, 1.68, -0.16, 0., 0.],\
                            [-0.07, -0.149, -0.16, 0.167, 0., 0.],\
                            [0., 0., 0., 0., 1., 0.], \
                            [0., 0., 0., 0., 0., 1.]], \
                    size=5000)
        tplib = np.clip(tplib, calparams.mins, calparams.maxs)
        plib = tplib * 0.
        plib[:, [0, 2, 3, 5]] = np.exp(tplib[:, [0, 2, 3, 5]])
        plib[:, 3] += 0.49
        plib[:, [1, 4]] = np.sinh(tplib[:, [1, 4]])

        # Instanciate calibration
        super(CalibrationGR6J, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib, \
            objfun_kwargs=objfun_kwargs)


