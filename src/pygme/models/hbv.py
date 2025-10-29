import numpy as np

from hydrodiy.stat import sutils
from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels

HBV_TMEAN = np.arcsinh([0.5, 300, 10, 1, 10,
                        100, 50, 4, 15, 20])

HBV_TCOV = np.eye(len(HBV_TMEAN))


def hbv_trans2true(x):
    return np.sinh(x)


def hbv_true2trans(x):
    return np.arcsinh(x)


class HBV(Model):
    def __init__(self):
        config = Vector(['nothing'], [0], [0], [1])

        vect = Vector(['LPRAT', 'FC', 'BETA', 'K0', 'K1', 'K2',
                       'LSUZ', 'CPERC', 'BMAX', 'CROUTE'],
                      [0.9, 100, 3.3, 0.5, 9, 105, 50, 2, 10, 26.5],
                      [0, 0, 0, 0, 2, 30, 1, 0, 0, 0],
                      [1, 600, 20, 2, 30, 250, 100, 8, 30, 50])

        params = ParamsVector(vect)

        states = Vector(['MOIST', 'SUZ', 'SLZ'])

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

        nuh = c_pygme_models_hydromodels.hbv_get_maxuh()
        self._dquh = np.zeros(nuh, dtype=np.float64)

    def initialise_fromdata(self):
        ''' Initialisation of HBV using
            * Production store:
            * Routing store:
        '''
        MOIST0 = self.params.FC / 2
        SUZ0 = self.params.LSUZ / 2
        SLZ0 = self.params.LSUZ / 2
        self.initialise([MOIST0, SUZ0, SLZ0])

    def run(self):
        # Run hbv c code
        ierr = c_pygme_models_hydromodels.hbv_run(self.istart,
                                                  self.iend,
                                                  self.params.values,
                                                  self._dquh,
                                                  self.inputs,
                                                  self.states.values,
                                                  self.outputs)
        if ierr > 0:
            errmsg = "c_pygme_models_hydromodels.hbv_run"\
                     + f" returns {ierr}."
            raise ValueError(errmsg)


class CalibrationHBV(Calibration):
    def __init__(self, objfun=ObjFunBCSSE(0.5),
                 warmup=5*365,
                 timeit=False,
                 fixed=None,
                 objfun_kwargs={},
                 nparamslib=2000,
                 Pm=0, Em=0):
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

        # Initialisation arguments
        initial_kwargs = {}

        # Instanciate calibration
        super(CalibrationHBV, self).__init__(calparams,
                                             objfun=objfun,
                                             warmup=warmup,
                                             timeit=timeit,
                                             objfun_kwargs=objfun_kwargs,
                                             initial_kwargs=initial_kwargs)

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tplib = sutils.lhs_norm(nparamslib, HBV_TMEAN, HBV_TCOV)

        # Back transform
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = hbv_trans2true(tplib[i, :])
        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib
