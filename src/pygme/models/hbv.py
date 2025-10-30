import numpy as np

from hydrodiy.stat import sutils
from hydrodiy.data.containers import Vector
from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

import c_pygme_models_hydromodels

# Default from R code
# [0.9, 100, 3.3, 0.5, 9, 105, 50, 2, 10, 26.5],
HBV_PARAMS_DEFAULT = np.array([
    0.60,  # LPRAT
    394.28,  # FC
    4.44,  # BETA
    1.02,  # K0
    5.18,  # K1
    139.31,  # K2
    26.42,  # LSUZ
    0.82,  # CPERC
    1.77,  # BMAX
    26.97  # CROUTE
    ])

HBV_PARAMS_MINS = np.array([
    0.00,   # LPRAT
    2.46,   # FC
    0.00,   # BETA
    0.00,   # K0
    2.00,   # K1
    30.00,  # K2
    1.00,   # LSUZ
    0.00,   # CPERC
    0.00,   # BMAX
    0.00,   # CROUTE
    ])

HBV_PARAMS_MAXS = np.array([
    1.00,    # LPRAT
    600.00,  # FC
    20.00,   # BETA
    2.00,    # K0
    30.00,   # K1
    250.00,  # K2
    100.00,  # LSUZ
    8.00,    # CPERC
    27.97,   # BMAX
    50.00,   # CROUTE
    ])


HBV_TMEAN = np.array([0.55, 6.54, 1.95, 0.84, 2.18, 5.31, 3.35,
                      0.51, 1.04, 3.48])

HBV_TCOV = np.array([
                    [0.09, -0.095, 0.112, 0.012, -0.037, -0.099,
                     -0.057, 0.034, 0.033, 0.024],
                    [-0.095, 0.33, -0.1, -0.03, 0.1, 0.18, 0.03,
                     -0.14, -0.11, -0.09],
                    [0.112, -0.1, 0.47, 0.03, -0.12, 0.01, -0.19,
                     -0.03, 0.03, 0.01],
                    [0.012, -0.03, 0.03, 0.16, -0.05, -0.05, -0.03,
                     -0.01, -0.02, -0.02],
                    [-0.037, 0.1, -0.12, -0.05, 0.32, 0.07, 0.14,
                     -0.04, -0.07, -0.1],
                    [-0.099, 0.18, 0.01, -0.05, 0.07, 0.74, -0.04,
                     -0.12, -0.11, -0.14],
                    [-0.057, 0.03, -0.19, -0.03, 0.14, -0.04, 1.5,
                     0.33, 0.09, -0.1],
                    [0.034, -0.14, -0.03, -0.01, -0.04, -0.12, 0.33,
                     0.49, 0.11, 0.02],
                    [0.033, -0.11, 0.03, -0.02, -0.07, -0.11, 0.09,
                     0.11, 0.61, 0.34],
                    [0.024, -0.09, 0.01, -0.02, -0.1, -0.14, -0.1,
                     0.02, 0.34, 1.7]
                    ])


def hbv_trans2true(x):
    return np.clip(np.sinh(x), HBV_PARAMS_MINS, HBV_PARAMS_MAXS)


def hbv_true2trans(x):
    return np.arcsinh(x)


class HBV(Model):
    def __init__(self):
        config = Vector(['nothing'], [0], [0], [1])

        vect = Vector(['LPRAT', 'FC', 'BETA', 'K0', 'K1', 'K2',
                       'LSUZ', 'CPERC', 'BMAX', 'CROUTE'],
                      HBV_PARAMS_DEFAULT,
                      HBV_PARAMS_MINS,
                      HBV_PARAMS_MAXS)
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

        self.paramslib = plib
