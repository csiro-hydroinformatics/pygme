import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector, UH
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels


class GR4J(Model):

    def __init__(self):

        # Config vector
        config = Vector(['continuous'],\
                    [0], [0], [1])

        # params vector
        vect = Vector(['X1', 'X2', 'X3', 'X4'], \
                    [400, -1, 50, 0.5], \
                    [10, -50, 1, 0.5], \
                    [2e4, 50, 5e3, 1e2])
        uhs = [UH('gr4j_ss1_daily', 3), UH('gr4j_ss2_daily', 3)]
        params = ParamsVector(vect, uhs)

        # State vector
        states = Vector(['S', 'R'])

        # Model
        super(GR4J, self).__init__('GR4J',
            config, params, states, \
            ninputs=2, \
            noutputsmax=9)

    def run(self):
        uh1 = self.params.uhs[0]
        uh2 = self.params.uhs[1]

        ierr = c_pygme_models_hydromodels.gr4j_run(uh1.nuh, \
            uh2.nuh, self.istart, self.iend, \
            self.params.values, \
            uh1.ord, \
            uh2.ord, \
            self.inputs, \
            uh1.states, \
            uh2.states, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.gr4j_run' +
                ' returns {0}').format(ierr))


class CalibrationGR4J(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, timeit=False):

        # Input objects for Calibration class
        model = GR4J()
        params = model.params


        trans2true = lambda x: np.array([
                        math.exp(x[0]), \
                        math.sinh(x[1]), \
                        math.exp(x[2]), \
                        0.49+math.exp(x[3])
                    ])

        true2trans = lambda x: np.array([
                        math.log(x[0]), \
                        math.asinh(x[1]), \
                        math.log(x[2]), \
                        math.log(x[3]-0.49)
                    ])

        cp = Vector(['tX1', 'tX2', 'tX3', 'tX4'], \
                mins=true2trans(params.mins),
                maxs=true2trans(params.maxs),
                defaults=true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=trans2true, \
            true2trans=true2trans)

        # Build parameter library from
        # MVT norm in transform space
        tplib = np.random.multivariate_normal(\
                    mean=[5.8, -0.78, 3.39, 0.86],
                    cov = [[1.16, 0.2, -0.15, -0.07],
                            [0.2, 1.79, -0.24, -0.149],
                            [-0.15, -0.24, 1.68, -0.16],
                            [-0.07, -0.149, -0.16, 0.167]],
                    size=2000)
        tplib = np.clip(tplib, calparams.mins, calparams.maxs)
        plib = tplib * 0.
        plib[:, [0, 2, 3]] = np.exp(tplib[:, [0, 2, 3]])
        plib[:, 3] += 0.49
        plib[:, 1] = np.sinh(tplib[:, 1])

        # Instanciate calibration
        super(CalibrationGR4J, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib)


