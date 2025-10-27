import math
import numpy as np

from hydrodiy.stat import sutils
from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels

HBV_TMEAN = np.array([5.8, -0.78, 3.39, 0.86, 0., 3.])

HBV_TCOV = np.array([[1.16, 0.2, -0.15, -0.07, 0., 0.],
                     [0.2, 1.79, -0.24, -0.149, 0., 0.],
                     [-0.15, -0.24, 1.68, -0.16, 0., 0.],
                     [-0.07, -0.149, -0.16, 0.167, 0., 0.],
                     [0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 1.]
                     ])


def hbv_trans2true(x):
    return np.sinh(x)


def hbv_true2trans(x):
    return np.arcsinh(x)


class HBV(Model):

    def __init__(self):

        # Config vector
        config = Vector(['nothing'], [0], [0], [1])

        # params vector
        vect = Vector(['LPRAT', 'FC', 'BETA', 'K0', 'K1', 'K2',
                       'LSUZ', 'CPERC', 'BMAX', 'CROUTE'],
                      [0.9, 100, 3.3, 0.5, 9, 105, 50, 2, 10, 26.5],
                      [0, 0, 0, 0, 2, 30, 1, 0, 0, 0],
                      [1, 600, 20, 2, 30, 250, 100, 8, 30, 50])

        params = ParamsVector(vect)

        # State vector
        states = Vector(['MOIST', 'SUZ', 'SLZ'])

        # Model
        super(HBV, self).__init__('HBV',
                                  config,
                                  params,
                                  states,
                                  ninputs=2,
                                  noutputsmax=13)

        self.outputs_names = ['Q', 'BRT', 'DQ',
                              'DMOIST', 'ETA', 'Q0',
                              'Q1', 'Q2', 'QG', 'SUM',
                              'MOIST', 'SUZ', 'SLZ'
                              ]

    def initialise_fromdata(self):
        ''' Initialisation of HBV using
            * Production store:
            * Routing store:
        '''
        MOIST0 = 0.
        SUZ0 = 0.
        SLZ0 = 0.
        self.initialise([MOIST0, SUZ0, SLZ0])

    def run(self):
        # Run hbv c code
        ierr = c_pygme_models_hydromodels.hbv_run(self.istart,
                                                  self.iend,
                                                  self.params.values,
                                                  self.inputs,
                                                  self.states.values,
                                                  self.outputs)
        if ierr > 0:
            errmsg = "c_pygme_models_hydromodels.hbv_run"\
                     + f" returns {ierr}."
            raise ValueError(errmsg)


class CalibrationHBV(Calibration):

    def __init__(self,
                 objfun=ObjFunBCSSE(0.2),
                 warmup=5*365,
                 timeit=False,
                 fixed=None,
                 objfun_kwargs={}):

        # Input objects for Calibration class
        model = HBV()
        params = model.params

        cp = Vector(['tLPRAT', 'tFC', 'tBETA', 'tK0', 'tK1', 'tK2',
                     'tLSUZ', 'tCPERC', 'tBMAX', 'tCROUTE'],
                    mins=hbv_true2trans(params.mins),
                    maxs=hbv_true2trans(params.maxs),
                    defaults=hbv_true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp,
                                      trans2true=hbv_trans2true,
                                      true2trans=hbv_true2trans,
                                      fixed=fixed)

        # Build parameter library from
        # MVT norm in transform space
        tplib = sutils.lhs_norm(5000, HBV_TMEAN, HBV_TCOV)
        tplib = np.clip(tplib, calparams.mins, calparams.maxs)
        plib = tplib * 0.
        plib[:, [0, 2, 3, 5]] = np.exp(tplib[:, [0, 2, 3, 5]])
        plib[:, 3] += 0.49
        plib[:, [1, 4]] = np.sinh(tplib[:, [1, 4]])

        # Initialisation arguments
        initial_kwargs = {}

        # Instanciate calibration
        super(CalibrationHBV, self).__init__(calparams,
                                             objfun=objfun,
                                             warmup=warmup,
                                             timeit=timeit,
                                             paramslib=plib,
                                             objfun_kwargs=objfun_kwargs,
                                             initial_kwargs=initial_kwargs())
