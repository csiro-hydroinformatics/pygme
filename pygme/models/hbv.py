import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector, UH
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels


class HBV(Model):

    def __init__(self):

        # Config vector
        config = Vector(['nothing'],\
                    [0], [0], [1])

        # params vector
        vect = Vector(
                    ['LPRAT', 'FC', 'BETA', 'K0', 'K1', 'K2', \
                        'LSUZ', 'CPERC', 'BMAX', 'CROUTE'], \
                    [100, 2.2, 0.5, 9, 105, 50, 2, 10, 26.5], \
                    [0, 0, 0, 0, 2, 30, 1, 0, 0, 0], \
                    [1, 600, 20, 2, 30 ,250, 100, 8, 30, 50])

        params = ParamsVector(vect)

        # State vector
        states = Vector(['MOIST', 'SUZ', 'SLZ'])

        # Model
        super(HBV, self).__init__('HBV',
            config, params, states, \
            ninputs=2, \
            noutputsmax=10)

        self.outputs_names = ['Q', 'BRT', 'DQ', \
                    'DMOIST', 'ETA', 'Q0', 'Q1', 'Q2', 'QG', 'SUM']


    def run(self):
        # Run hbv c code
        ierr = c_pygme_models_hydromodels.hbv_run(
            self.istart, self.iend, \
            self.params.values, \
            self.inputs, \
            self.states.values, \
            self.outputs)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.hbv_run' +
                ' returns {0}').format(ierr))


class CalibrationHBV(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.2), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    objfun_kwargs={}):

        # Input objects for Calibration class
        model = HBV()
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

        cp = Vector(['tLPRAT', 'tFC', 'tBETA', 'tK0', 'tK1', 'tK2', \
                        'tLSUZ', 'tCPERC', 'tBMAX', 'tCROUTE'], \
                mins=true2trans(params.mins),
                maxs=true2trans(params.maxs),
                defaults=true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=trans2true, \
            true2trans=true2trans,\
            fixed=fixed)

        # Build parameter library from
        # MVT norm in transform space
        # TODO !!!!!!!
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
        super(CalibrationHBV, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            paramslib=plib, \
            objfun_kwargs=objfun_kwargs)


