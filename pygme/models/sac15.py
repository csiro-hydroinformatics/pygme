import math
import numpy as np
import pandas as pd

from hydrodiy.data.containers import Vector
from hydrodiy.stat import sutils

from pygme.model import Model, ParamsVector
from pygme.calibration import Calibration, \
                    CalibParamsVector, \
                    ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels
    import c_pygme_models_utils

# Transformed parameters mean and covariance
SAC15_TMEAN = np.ones(15)
SAC15_TCOV = np.eye(15)

# Transformation functions for sac parameters
def sac15_trans2true(tx):
    x = np.exp(tx)
    x[0] = tx[0]                   # ADIMP
    x[3] = 1./(1.+math.exp(-tx[3]))# LZPK
    x[4] = 1./(1.+math.exp(-tx[4]))# LZSK
    x[6] = (9.99+tx[6])/19.98      # PFREE
    x[8] = tx[8]                   # SARVA
    x[9] = math.sinh((5+tx[9])/10.)# SIDE
    x[12] = 1./(1.+math.exp(-tx[12]))# UZK
    return x

def sac15_true2trans(x):
    tx = np.log(np.maximum(1e-10, x))
    tx[0] = x[0]                    # ADIMP
    tx[3] = math.log(x[3]/(1-x[3])) # LZPK
    tx[4] = math.log(x[4]/(1-x[4])) # LZSK
    tx[6] = x[6]*19.98-9.99         # PFREE
    tx[8] = x[8]                    # SARVA
    tx[9] = 10.*math.asinh(x[9])-5. # SIDE
    tx[12] = math.log(x[12]/(1-x[12])) # UZK

    return tx


class SAC15(Model):

    def __init__(self):
        config = Vector(["nodata"], [0], [0], [1])

        # Param vector
        vect = Vector(['Adimp', 'Lzfpm', 'Lzfsm', 'Lzpk', \
                    'Lzsk', 'Lztwm', 'Pfree', 'Rexp', 'Sarva', \
                    'Side', 'Lag', 'Uzfwm', 'Uzk', 'Uztwm', 'Zperc'], \
                    defaults=[0.1, 82., 32., 0.04, 0.24, 179.,
                            0.4, 1.81, 0.01, 0., 0., 49., 0.4, 76., 60.], \
                    mins=[1e-5, 1e-2, 1e-2, 1e-3, 1e-3, 10.,
                            1e-2, 1., 0., -0.2, 0, 1e-1, 1e-5, 1., 1e-2], \
                    maxs=[0.9, 5e2, 5e2, 0.9, 0.9, 1e3, 0.5,
                            6., 0.5, 0.5, 100., 1e3, 1-1e-10, 1e3, 1e3])
        params = ParamsVector(vect)

        # UH
        params.add_uh("lag", lambda params: params.Lag)

        # State vector
        states = Vector(["Uztwc", "Uzfwc", "Lztwc", \
                    "Lzfsc", "Lzfpc", "Adimc"], check_bounds=False)

        # Model
        super(SAC15, self).__init__("SAC15", \
                config, params, states, \
                ninputs=2, \
                noutputsmax=11)

    def initialise_fromdata(self):
        # Initialise stores
        Lzfpm = self.params.Lzfpm
        Lzfsm = self.params.Lzfsm
        Lztwm = self.params.Lztwm
        Uzfwm = self.params.Uzfwm
        Uztwm = self.params.Uztwm

        self.initialise(states= [Uztwm, Uzfwm/2, Lztwm/2,
                Lzfsm/2, Lzfpm/2, Uztwm+Lztwm/2])

    def run(self):
        # Get uh object
        _, uh = self.params.uhs[0]

        ierr = c_pygme_models_hydromodels.sac15_run(uh.nord,\
            self.istart, self.iend,\
            self.params.values,\
            uh.ord,\
            self.inputs, \
            uh.states, \
            self.states.values,
            self.outputs)

        if ierr > 0:
            raise ValueError(('c_pygme_models_hydromodels.sac15_run' +
                ' returns {0}').format(ierr))



class CalibrationSAC15(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.5), \
                    warmup=5*365, \
                    timeit=False, \
                    fixed=None, \
                    objfun_kwargs={}, \
                    nparamslib=2000):

        # Input objects for Calibration class
        model = SAC15()
        params = model.params

        # Calib param vector
        cp = Vector(['tAdimp', 'tLzfpm', 'tLzfsm', 'tLzpk', \
                    'tLzsk', 'tLztwm', 'tPfree', 'tRexp', 'tSarva', \
                    'tSide', 'tLag', 'tUzfwm', 'tUzk', 'tUztwm', 'tZperc'], \
                        mins=sac15_true2trans(params.mins),
                        maxs=sac15_true2trans(params.maxs),
                        defaults=sac15_true2trans(params.defaults))

        calparams = CalibParamsVector(model, cp, \
            trans2true=sac15_trans2true, \
            true2trans=sac15_true2trans,\
            fixed=fixed)

        # Instanciate calibration
        super(CalibrationSAC15, self).__init__(calparams, \
            objfun=objfun, \
            warmup=warmup, \
            timeit=timeit, \
            objfun_kwargs=objfun_kwargs)

        # Build parameter library from
        # MVT norm in transform space using latin hypercube
        tplib = sutils.lhs_norm(nparamslib, SAC15_TMEAN, SAC15_TCOV)

        # Back transform
        plib = tplib * 0.
        for i in range(len(plib)):
            plib[i, :] = sac15_trans2true(tplib[i, :])
        plib = np.clip(plib, model.params.mins, model.params.maxs)
        self.paramslib = plib




